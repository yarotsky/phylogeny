from __future__ import print_function

'''
TODO: measurement of objective function for original tree
TODO: metrics of tree reconstruction performance (distance between true and reconstructed tree)
TODO: arbitrary Nbases (currently only 0/1)
TODO: the standard simple example from MEGA 
TODO: split restriction to prevent excessive tree growth
TODO: cuda version
'''
#from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import torch
#from torch.autograd import grad
import networkx as nx
from cvxopt import matrix, solvers
from Cython.Compiler.Main import verbose
#import cvxopt


#====== SETTINGS =============

eps = 1e-6

#deviceType = 'cuda' # not yet supported
deviceType = 'cpu'

if deviceType == 'cuda': # not yet supported
    print ('Using GPU\n')
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    dtype = torch.float32
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
        
elif deviceType == 'cpu':
    print ('Using CPU\n')
    device = torch.device('cpu')
    dtype = torch.float64    
    torch.set_default_tensor_type(torch.DoubleTensor)
    
#==========================


def getSpecies(Nspecies=5, 
               case='basic0', 
               seed=0, 
               N1=None # number of species in one family for 'two_families' case
               ):
    '''Generate a set of named genomes (species)'''
        
    if case == 'book':
        assert Nspecies <= 5
        Alpha = [1, 0, 0, 1, 1, 0]
        Beta = [0, 0, 1, 0, 0, 0]
        Gamma = [1, 1, 0, 0, 0, 0]
        Delta = [1, 1, 0, 1, 1, 1]
        Epsilon = [0, 0, 1, 1, 1, 0]
        
        speciesNames = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'][:Nspecies]
        speciesData = np.array([locals()[name] for name in speciesNames])
    
    elif case == 'basic0':
        assert Nspecies <= 5
        Alpha = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        Beta = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        Gamma = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Delta = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Epsilon = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        speciesNames = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'][:Nspecies]
        speciesData = np.array([locals()[name] for name in speciesNames])

    elif case == 'basic':
        assert Nspecies <= 5
        Alpha = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Beta = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Gamma = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Delta = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Epsilon = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        speciesNames = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'][:Nspecies]
        speciesData = np.array([locals()[name] for name in speciesNames])
        
        
    elif case == 'two_families':
        np.random.seed(seed)
        M = 1000 # genome size
        N2 = Nspecies-N1
        Nspecies = N1+N2
        root = np.array([np.zeros(M), np.ones(M)]).astype(int).ravel()
        print(('Original species:', root)) 
        assert root.shape == (2*M,)
        t = 1.
        t1 = 0.2
        t2 = 0.1
        family1 = root ^ (np.random.rand(2*M) > 0.5*(1+np.exp(-t1))) 
        print (family1)
        family2 = root ^ (np.random.rand(2*M) > 0.5*(1+np.exp(-t2)))
        print (family2)
        k = 0
        speciesNames = []
        speciesData = []
        print ('============= 1')
        for k in range(N1):
            name = 'sp'+str(k)
            speciesNames.append(name)
            speciesData.append(family1 ^ (np.random.rand(2*M) > 0.5*(1+np.exp(-(t-t1)))))
            print((speciesData[-1]))
        print ('============= 2')
        for k in range(N1, N1+N2):
            name = 'sp'+str(k)
            speciesNames.append(name)
            speciesData.append(family2 ^ (np.random.rand(2*M) > 0.5*(1+np.exp(-(t-t2)))))  
            print((speciesData[-1]))  
            
        speciesData = np.array(speciesData)
          
    return speciesData, speciesNames


def testgetSpecies():
    speciesData, speciesNames = getSpecies(Nspecies=5, N1=2, case='two_families')
    print(('speciesData:', speciesData))
    print(('speciesNames:', speciesNames))
    
    
def genRandEvolTree(totalTime=1.,  # time from original to final species
                    genomesize=100, 
                    splitRate=2.,  # rate of branching into different species
                    mutationRate=0.1): # rate of mutation for one species and one genome position
    
    ''' Generate a random ground truth evolution tree'''
    
    G = nx.DiGraph() # the evolution tree
    G.batchsize = genomesize
    
    # create root node
    G.add_node(-1, # node label
               t=0., # time
               level=0, # number of splits from root
               ypos=0.5, # used only for plotting
               leaf=True, # a leaf of the current tree
               genome=np.zeros((G.batchsize,)).astype(int))
    
    # create tree nodes
    minTime = 0. # time of earliest node to be split
    minTimeNode = -1 # earliest node to be split
    while minTime < totalTime:    
        # first species after splitting   
        dt1 = np.random.exponential()/splitRate 
        if  G.nodes[minTimeNode]['t']+dt1 > totalTime:
            dt1 = totalTime-G.nodes[minTimeNode]['t']                 
        G.add_node(-len(G.nodes)-1, # for now, label with temporary negative numbers; will relabel later 
                   t=G.nodes[minTimeNode]['t']+dt1,
                   level=G.nodes[minTimeNode]['level']+1,
                   ypos=G.nodes[minTimeNode]['ypos']-2**(-G.nodes[minTimeNode]['level']-2),
                   leaf=True,
                   genome=G.nodes[minTimeNode]['genome'] ^ (np.random.rand(genomesize) > 0.5*(1+np.exp(-mutationRate*dt1))))
        G.add_edge(minTimeNode, -len(G.nodes), w=torch.tensor(1.))
        
        # second species after splitting 
        dt2 = np.random.exponential()/splitRate   
        if  G.nodes[minTimeNode]['t']+dt2 > totalTime:
            dt2 = totalTime-G.nodes[minTimeNode]['t']            
        G.add_node(-len(G.nodes)-1,
                   t=G.nodes[minTimeNode]['t']+dt2,
                   level=G.nodes[minTimeNode]['level']+1,
                   ypos=G.nodes[minTimeNode]['ypos']+2**(-G.nodes[minTimeNode]['level']-2),
                   leaf=True,
                   genome=G.nodes[minTimeNode]['genome'] ^ (np.random.rand(genomesize) > 0.5*(1+np.exp(-mutationRate*dt2))))
        G.add_edge(minTimeNode, -len(G.nodes), w=torch.tensor(1.))
        
        # the split node is no longer a leaf
        G.nodes[minTimeNode]['leaf'] = False
        
        # find new earliest node to split
        minTimeNode = None
        minTime = np.inf
        for node in G.nodes:
            if G.nodes[node]['leaf']:
                if G.nodes[node]['t'] < minTime:
                    minTimeNode = node
                    minTime = G.nodes[node]['t']    
        
    speciesInds = [node for node in G.nodes if G.nodes[node]['t'] == totalTime]
    nonSpeciesInds = [node for node in G.nodes if G.nodes[node]['t'] < totalTime]
    assert 2*len(speciesInds)-1 == len(G.nodes)
    speciesSorted = np.argsort([G.nodes[species]['ypos'] for species in speciesInds]) 
    G.speciesNames = ['sp'+str(n+1) for n in range(len(speciesInds))]
    speciesIndsSorted = [speciesInds[ind] for ind in speciesSorted]    
    nonSpeciesSorted = np.argsort([G.nodes[species]['t'] for species in nonSpeciesInds])
    nonSpeciesIndsSorted = [nonSpeciesInds[k] for k in nonSpeciesSorted]
    nonSpeciesNames = list(range(len(nonSpeciesSorted)))
    D = list(zip(speciesIndsSorted+nonSpeciesIndsSorted, G.speciesNames+nonSpeciesNames))
    
    nx.relabel_nodes(G, dict(D), copy=False)
    G.Nspecies = len(G.speciesNames)    
         
    return G
    
    
def testGenRandEvolTree():
    Gtrue = genRandEvolTree(totalTime=0.5)
    print(('Nodes:', Gtrue.nodes))
    for species in Gtrue.speciesNames:
        print((species, Gtrue.nodes[species]))
    plot(Gtrue, nonuniformSpecies=True)
    

def getGraph(speciesData, 
             speciesNames, 
             mode=0,   # one of several versions how to store and process SNP information   
             verbose=False
             ): 
    
    '''Generate initial graph for given set of species'''
    
    Nspecies = len(speciesNames) 
    G = nx.DiGraph()
    G.mode = mode
    G.batchsize = speciesData.shape[1]
    
    # species nodes
    for name in speciesNames:
        if verbose:
            print('-----', name)
        prob = np.array(speciesData[speciesNames.index(name)])
        if verbose:
            print(prob)
        
        initA = np.empty((G.batchsize,2))
        initA[:,0] = 1-eps-prob*(1-2*eps)
        initA[:,1] = eps+prob*(1-2*eps)
        if mode == 1:
            G.add_node(name, 
                       t=torch.tensor(0., requires_grad=False),
                       snp=torch.tensor(np.log(initA), requires_grad=False))
        elif mode == 2:
            G.add_node(name, 
                       t=torch.tensor(0., requires_grad=False),
                       snp=torch.tensor(initA, requires_grad=False))  
            
        elif mode == 0:
            G.add_node(name, 
                       t=torch.tensor(0., requires_grad=False),
                       snp=torch.tensor(initA, requires_grad=False))           
        if verbose:
            print(G.nodes[name])
    
    # initial node (original species) 
    G.add_node(0, 
               t=torch.tensor(-1., requires_grad=True),
               level=0,
               ypos=0.5,
               width=1.,
               snp=torch.tensor(np.ones((G.batchsize,2)), requires_grad=False))
    
    for node in speciesNames:
        G.add_edge(0, node, 
                   w=torch.tensor(1., requires_grad=True))
    
    # special ("unphysical") node preceding all other nodes
    G.add_node('out', width=1., snp=torch.tensor(np.ones((G.batchsize,2)), requires_grad=False))
    G.add_edge('out', 0, w=torch.tensor(1., requires_grad=False))
                     
    G.speciesNames = speciesNames
    G.NinternalNodes = 1
    G.Nspecies = Nspecies
        
    return G


def testGraph():
    speciesData, speciesNames = getSpecies(Nspecies=5, case='basic')
    G = getGraph(speciesData, speciesNames, verbose=True)
    print(G)
    plot(G)
    
    
def optPosPlot(G, 
               retain=False  # retain information about successors etc. for future use
               ):
    
    '''Plot optimization: optimize positions of nodes to make plots as uncluttered as possible.
    
    The tree shouldn not have leaves other than species.'''
    
    internalNodes = [n for n in G.nodes if not (n in G.speciesNames or n == 'out')]
    indSorted = np.argsort([-float(G.nodes[n]['t'].data) for n in internalNodes])
    nodesSorted = [internalNodes[ind] for ind in indSorted]
    
    for node in nodesSorted:
        assert len(getNonSpeciesLeaves(G)) == 0, 'Leaves must be species only!'
        # immediate successors
        G.nodes[node]['successors'] = [n for n in G.successors(node) if n in range(G.NinternalNodes)] # not including species
        G.nodes[node]['trueSuccessors'] = []
        G.nodes[node]['predecessors'] = [n for n in G.predecessors(node) if n in range(G.NinternalNodes)] 
        
        # all successors
        G.nodes[node]['successorsAll'] = []+G.nodes[node]['successors']
        for n in G.nodes[node]['successors']:
            G.nodes[node]['successorsAll'].extend(G.nodes[n]['successorsAll'])
                        
        G.nodes[node]['sumW0'] = 0
        G.nodes[node]['sumW1'] = 0
        
        if not G.nodes[node]['successors']: # the node only has species as successors
            for n in G.successors(node):
                if n in G.speciesNames:
                    G.nodes[node]['sumW0'] += float(G.edges[(node, n)]['w'].data)
                    G.nodes[node]['sumW1'] += float(G.edges[(node, n)]['w'].data)*(G.nodes[node]['ypos']-
                                                                            float(G.speciesNames.index(n))/G.Nspecies)
        else:
            for n in G.nodes[node]['successors']:
                G.nodes[node]['sumW0'] += G.nodes[n]['sumW0']*float(G.edges[(node,n)]['w'].data)
                G.nodes[node]['sumW1'] += G.nodes[n]['sumW1']*float(G.edges[(node,n)]['w'].data)
            for n in G.successors(node):
                if n in G.speciesNames:
                    G.nodes[node]['sumW0'] += float(G.edges[(node, n)]['w'].data)
                    G.nodes[node]['sumW1'] += float(G.edges[(node, n)]['w'].data)*(G.nodes[node]['ypos']-
                                                                            float(G.speciesNames.index(n))/G.Nspecies)
    
    for node in nodesSorted:
        if G.nodes[node]['sumW0'] == 0:
            plot(G) 
        if G.nodes[node]['predecessors']:
            assert np.abs(np.sum([float(G.edges[(n1,node)]['w'].data) for n1 in G.nodes[node]['predecessors']])-1.) < 3*eps
            k = np.argmax([float(G.edges[(n1,node)]['w'].data) for n1 in G.nodes[node]['predecessors']])
            n1 = G.nodes[node]['predecessors'][k]
            G.nodes[n1]['trueSuccessors'].append(node)    
    
    for node in nodesSorted[::-1]:                    
        if G.nodes[node]['trueSuccessors']:
            optPosL = [G.nodes[n]['ypos']-G.nodes[n]['sumW1']/G.nodes[n]['sumW0'] for n in G.nodes[node]['trueSuccessors']]
            indSorted = np.argsort(optPosL)
            for i, ind in enumerate(indSorted):
                y = G.nodes[node]['ypos']-G.nodes[node]['width']/2.+G.nodes[node]['width']*(0.3+0.4*np.random.rand()+i)/len(G.nodes[node]['trueSuccessors'])
                n = G.nodes[node]['trueSuccessors'][ind]
                dy = y-G.nodes[n]['ypos']
                G.nodes[n]['width'] = G.nodes[node]['width']/len(G.nodes[node]['trueSuccessors'])
                for n1 in G.nodes[n]['successorsAll']+[n]:
                    G.nodes[n1]['ypos'] += dy  
                    G.nodes[n1]['sumW1'] += G.nodes[n1]['sumW0']*dy 
                                       
    if not retain:
        for node in nodesSorted:
            del G.nodes[node]['successors']
            del G.nodes[node]['trueSuccessors']
            del G.nodes[node]['predecessors']
            del G.nodes[node]['successorsAll']
            del G.nodes[node]['sumW0']
            del G.nodes[node]['sumW1']    
  

def plot(G, 
         show=True, # show plot or only return plot object
         figNum=None, # if provided, add plot to existing figure
         nonuniformSpecies=False # try to distribute species uniformly on y axis
         ):
    '''Plot the phylogenetic tree'''
    
    if figNum is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figNum)
    pos = nx.spring_layout(G, iterations=100)
    for node in pos:
        if node in G.speciesNames:
            if nonuniformSpecies:
                pos[node][1] = G.nodes[node]['ypos']
            else:
                pos[node][1] = float(G.speciesNames.index(node))/G.Nspecies
        elif node == 'out':
            pos[node][1] = 0.5
        else:
            pos[node][1] = G.nodes[node]['ypos']
        if 't' in list(G.nodes[node].keys()):
            pos[node][0] = G.nodes[node]['t']
            if not isinstance(pos[node][0], float):
                pos[node][0] = pos[node][0].data
        else:
            pos[node][0] = 0.
    minT = np.min([pos[node][0] for node in pos])
    if 'out' in G.nodes:
        pos['out'][0] = minT-0.01
    
    # edge color depends on its 'w' value
    edge_color=[1.-G.edges[edge]['w'].data for edge in G.edges]

    nx.draw(G, pos, with_labels=True, node_size=100, width=2, edge_color=edge_color, 
            edge_cmap=plt.cm.inferno, edge_vmin=0., edge_vmax=1.) 

    if show:
        plt.show() 
    return fig 


def getLikelihood(G):  
    ''' Compute tree likelihood associated with current edge weights''' 
    
    # sort internal nodes (not leaves) 
    indSorted = np.argsort([-G.nodes[n]['t'].data.numpy() for n in range(G.NinternalNodes)])
    batchsize = G.nodes['out']['snp'].shape[0]
    
    # for each internal node, define its SNP array using SNP's of successor nodes and edge weights
    for n, k in enumerate(indSorted):
        t = G.nodes[k]['t']
        
        # G.mode: one of several versions to compute likelihood; defailt is G.mode = 0 
        if G.mode == 1:      
            G.nodes[k]['snp'] = torch.tensor(np.ones((batchsize,2))*np.log(0.5))
            for node in G.speciesNames+indSorted[:n].tolist():
                t_ = G.nodes[node]['t']
                assert t_ >= t                                              
                if (k, node) in G.edges:  
                                                            
                    G.nodes[k]['snp'][:,0] += torch.log(torch.exp(G.nodes[node]['snp'][:,0]+
                                                             (G.edges[(k, node)]['w']*torch.log(0.5+0.5*torch.exp((t-t_)))
                                                              +(-G.edges[(k, node)]['w'])*np.log(0.5)))+
                                                         torch.exp(G.nodes[node]['snp'][:,1]+
                                                             (G.edges[(k, node)]['w']*torch.log(0.5-0.5*torch.exp((t-t_)))
                                                              +(-G.edges[(k, node)]['w'])*np.log(0.5)))
                                                        ) 
    
                    G.nodes[k]['snp'][:,1] += torch.log(torch.exp(G.nodes[node]['snp'][:,1]+
                                                             (G.edges[(k, node)]['w']*torch.log(0.5+0.5*torch.exp((t-t_)))
                                                              +(-G.edges[(k, node)]['w'])*np.log(0.5)))+
                                                        torch.exp(G.nodes[node]['snp'][:,0]+
                                                             (G.edges[(k, node)]['w']*torch.log(0.5-0.5*torch.exp((t-t_)))
                                                              +(-G.edges[(k, node)]['w'])*np.log(0.5)))
                                                        )        
        elif G.mode == 0:    
            G.nodes[k]['snp'] = torch.tensor(np.ones((batchsize,2))*(0.5))
            for node in G.speciesNames+indSorted[:n].tolist():
                t_ = G.nodes[node]['t']
                assert t_ >= t                                              
                if (k, node) in G.edges:  
                    w = G.edges[(k, node)]['w']
                    snp = G.nodes[node]['snp']
                    
                    if node in G.speciesNames:                                                            
                        G.nodes[k]['snp'][:,0] *= (snp[:,0]*(1.+torch.exp(t-t_))**w+
                                                   snp[:,1]*(1.-torch.exp(t-t_))**w)        
                        G.nodes[k]['snp'][:,1] *= (snp[:,0]*(1.-torch.exp(t-t_))**w+
                                                   snp[:,1]*(1.+torch.exp(t-t_))**w) 
                    else:
                        G.nodes[k]['snp'][:,0] *= (snp[:,0]*(1.+torch.exp(t-t_))+
                                                   snp[:,1]*(1.-torch.exp(t-t_)))**w        
                        G.nodes[k]['snp'][:,1] *= (snp[:,0]*(1.-torch.exp(t-t_))+
                                                   snp[:,1]*(1.+torch.exp(t-t_)))**w   

        elif G.mode == 2:    
            G.nodes[k]['snp'] = torch.tensor(np.ones((batchsize,2))*(0.5))
            for node in G.speciesNames+indSorted[:n].tolist():
                t_ = G.nodes[node]['t']
                assert t_ >= t                                              
                if (k, node) in G.edges:                                       
                    G.nodes[k]['snp'][:,0] *= ((G.nodes[node]['snp'][:,0]*(1.+torch.exp(t-t_)))**G.edges[(node, k)]['w']+
                                               (G.nodes[node]['snp'][:,1]*(1.-torch.exp(t-t_)))**G.edges[(node, k)]['w'])
    
                    G.nodes[k]['snp'][:,1] *= ((G.nodes[node]['snp'][:,0]*(1.-torch.exp(t-t_)))**G.edges[(node, k)]['w']+
                                               (G.nodes[node]['snp'][:,1]*(1.+torch.exp(t-t_)))**G.edges[(node, k)]['w']) 
            
    G.nodes['out']['snp'] = torch.tensor(np.zeros((batchsize,2)))

    if G.mode  == 1:
        G.nodes['out']['snp'] = G.nodes[0]['snp']        
        likelihood = torch.log(torch.exp(G.nodes['out']['snp']).sum(dim=1)).mean()
        
    elif G.mode == 2:
        G.nodes['out']['snp'] = G.nodes[0]['snp']        
        likelihood = torch.log(G.nodes['out']['snp'].sum(dim=1)).mean()    

    elif G.mode == 0:
        G.nodes['out']['snp'] = G.nodes[0]['snp']        
        likelihood = torch.log(G.nodes['out']['snp'].sum(dim=1)).mean()     
            
    return likelihood


def testGetLikelihood():
    speciesData, speciesNames = getSpecies(Nspecies=3, case='basic')
    G = getGraph(speciesData, speciesNames,  verbose=False) 
    plot(G)
    print(getLikelihood(G))
    
def testGetLikelihood2():
    speciesData, speciesNames = getSpecies(Nspecies=5, case='basic')
    G = getGraph(speciesData, speciesNames,  verbose=False) 
    print(getLikelihood(G))
    
    
def getObj(G):
    '''The objective function (the higher the better)'''
    return getLikelihood(G)


def serialize(G, 
              grad=False, # include gradient values
              printout=False, # print the parameter and its value
              gradRprop=False # instead of original grad values, output those used in Rprop
              ):
    '''Write all fitted tree parameters (node positions and edge weights) into a list'''
    c = [] 
    if not printout:  
        if grad:   
            for n in range(G.NinternalNodes): 
                c.append(float(G.nodes[n]['t'].grad.data))                    
                for n1 in G.speciesNames:
                    if (n, n1) in G.edges:
                        c.append(float(G.edges[(n,n1)]['w'].grad.data))
                for n1 in range(n):
                    if (n, n1) in G.edges:
                        c.append(float(G.edges[(n,n1)]['w'].grad.data))
        else:
            for n in range(G.NinternalNodes): 
                c.append(float(G.nodes[n]['t'].data))
                for n1 in G.speciesNames:
                    if (n, n1) in G.edges:
                        c.append(float(G.edges[(n,n1)]['w'].data))
                for n1 in range(n):
                    if (n, n1) in G.edges:
                        c.append(float(G.edges[(n,n1)]['w'].data))
        return c
    else:
        if grad:   
            for n in range(G.NinternalNodes): 
                if not gradRprop:
                    grd = G.nodes[n]['t'].grad
                else:
                    grd = G.nodes[n]['t_grad']               
                print('nodes['+str(n)+'][t]:', float(grd.data))
                for n1 in G.speciesNames:
                    if (n, n1) in G.edges:
                        if not gradRprop:
                            grd = G.edges[(n,n1)]['w'].grad
                        else:
                            grd = G.edges[(n,n1)]['w_grad'] 
                        print('edges[('+str(n)+','+str(n1)+')][w]', float(grd.data))
                for n1 in range(n):
                    if (n, n1) in G.edges:
                        if not gradRprop:
                            grd = G.edges[(n,n1)]['w'].grad
                        else:
                            grd = G.edges[(n,n1)]['w_grad'] 
                        print('edges[('+str(n)+','+str(n1)+')][w]', float(grd.data))
        else:
            for n in range(G.NinternalNodes): 
                print('nodes['+str(n)+'][t]:', float(G.nodes[n]['t'].data))
                for n1 in G.speciesNames:
                    if (n, n1) in G.edges:
                        print('edges[('+str(n)+','+str(n1)+')][w]', float(G.edges[(n,n1)]['w'].data))
                for n1 in range(n):
                    if (n, n1) in G.edges:
                        print('edges[('+str(n)+','+str(n1)+')][w]', float(G.edges[(n,n1)]['w'].data)) 
                                      
       
                
def deserialize(G, c): 
    '''Assign values from list c to tree parameters (inverse of serialize())'''   
    k = 0
    for n in range(G.NinternalNodes): 
        G.nodes[n]['t'] = torch.tensor(c[k], requires_grad=True)
        k += 1
        for n1 in G.speciesNames:
            if (n, n1) in G.edges:
                G.edges[(n,n1)]['w'] = torch.tensor(c[k], requires_grad=True)
                k += 1
        for n1 in range(n):
            if (n, n1) in G.edges:
                G.edges[(n,n1)]['w'] = torch.tensor(c[k], requires_grad=True)
                k += 1           


def testSerialize():     
    speciesData, speciesNames = getSpecies(Nspecies=5, case='basic')
    G = getGraph(speciesData, speciesNames, verbose=False) 
    obj = getObj(G)
    obj.backward()
    c0 = serialize(G, grad=True)
    print(len(c0))
    print(c0)
    serialize(G, grad=True, printout=True)
    c00 = serialize(G, grad=True)
    print(len(c00))
    print(c00)
    c1 = np.arange(len(c0)).astype('float')
    print(c1)
    deserialize(G, c1)
    c2 = np.array(serialize(G))
    assert np.linalg.norm(c1-c2) == 0 
    

def zeroGrad(G):
    '''Zero grad values for all trainable tree parameters'''
    for node in range(G.NinternalNodes):
        if G.nodes[node]['t'].grad:
            G.nodes[node]['t'].grad.data.zero_()
        for n in G.speciesNames+list(range(G.NinternalNodes))+['out']:
            if n == node:
                continue   
            if (node, n) in G.nodes:        
                G.edges[(node, n)]['w'].grad.data.zero_()


def prepareOptProblem(G, onlyC=False, minW=eps):
    '''Prepare optimization problem for CVXopt'''
    c = serialize(G, grad=True)      
    c = -matrix(c)  
    if onlyC:
        return c
    
    Aopt = np.zeros((len(G.nodes)-2, len(c)))
    b = np.zeros((len(G.nodes)-2,))  
    
    k = 0
    for n in range(G.NinternalNodes):
        k += 1        
        for n1, species in enumerate(G.speciesNames): 
            if (n, species) in G.edges:           
                Aopt[n1, k] = 1.
                b[n1] = 1.
                k += 1
        for n1 in range(n):
            if (n, n1) in G.edges:
                assert G.nodes[n1]['t'] > G.nodes[n]['t']
                if n != 0:
                    Aopt[G.Nspecies+n-1, k] = 1.
                    b[G.Nspecies+n-1] = 1.
                k += 1        
    
    Aopt = matrix(Aopt)    
    b = matrix(b)
    
    NinternalEdges = len([(n0,n1) for (n0,n1) in G.edges if not (n0 in G.speciesNames+['out'] or n1 in G.speciesNames+['out'])])      
    Gopt = np.zeros((2*len(c)+NinternalEdges, len(c)))
    h = np.zeros((2*len(c)+NinternalEdges,))
    k = 0
    k1 = 0
    node2k = []
    for n in range(G.NinternalNodes):
        node2k.append(k)
        Gopt[2*k, k] = 1.
        h[2*k] = 10.
        Gopt[2*k+1, k] = -1.
        h[2*k+1] = -eps
        k += 1        
        for n1 in G.speciesNames: 
            if (n, n1) in G.edges:           
                Gopt[2*k, k] = 1.
                h[2*k] = 1.+eps
                Gopt[2*k+1, k] = -1.
                h[2*k+1] = -minW
                k += 1
        for n1 in range(n):
            if (n, n1) in G.edges:
                Gopt[2*k, k] = 1.
                h[2*k] = 1.+eps
                Gopt[2*k+1, k] = -1.
                h[2*k+1] = -minW
                
                Gopt[2*len(c)+k1, node2k[n]] = 1.
                Gopt[2*len(c)+k1, node2k[n1]] = -1.
                h[2*len(c)+k1] = -eps
                k += 1
                k1 += 1
         
    assert k == len(c)
    assert k1 == NinternalEdges
                     
    Gopt = matrix(Gopt)
    h = matrix(h)
    
    primalstart = dict()
    primalstart['x'] = matrix(serialize(G))
    primalstart['s'] = h-Gopt*primalstart['x']
     
    return c, Aopt, b, Gopt, h, primalstart
   
    
def testPrepareOptProblem():
    speciesData, speciesNames = getSpecies(Nspecies=5, case='basic')
    G = getGraph(speciesData, speciesNames, verbose=False) 
    obj = getObj(G, 1)
    obj.backward()    
    c, Aopt, b, Gopt, h, primalstart = prepareOptProblem(G, onlyC=False)
    print(c) 
    print(Aopt) 
    print(b) 
    print(Gopt) 
    print(h)
    print(primalstart)
    

    

def split(G, 
          rand=0., # randomness in the redistributed edge weights 
          NspeciesThreshold=3, # only split if the node is connected to this many or more species
          ):
    
    '''Split tree nodes'''
    
    nNewNode = G.NinternalNodes
    for n in range(G.NinternalNodes):
        Nconnect2species = 0
        for n1 in G.successors(n):
            if n1 in G.speciesNames:
                Nconnect2species += 1
        if Nconnect2species >= NspeciesThreshold:          
            # TODO: don't split if std is small 
            G.add_node(nNewNode,
                       t=(0.8*G.nodes[n]['t'].clone().detach()).requires_grad_(True),
                       level=G.nodes[n]['level']+1,
                       ypos=G.nodes[n]['ypos']-2**(-G.nodes[n]['level']-2),
                       snp=torch.tensor(np.ones((G.batchsize,2)), requires_grad=False))
            nNewNode += 1
            G.add_node(nNewNode,
                       t=(0.8*G.nodes[n]['t'].clone().detach()).requires_grad_(True),
                       level=G.nodes[n]['level']+1,
                       ypos=G.nodes[n]['ypos']+2**(-G.nodes[n]['level']-2),
                       snp=torch.tensor(np.ones((G.batchsize,2)), requires_grad=False))
            nNewNode += 1
            G.add_edge(n, nNewNode-2, w=torch.tensor(1., requires_grad=True))  
            G.add_edge(n, nNewNode-1, w=torch.tensor(1., requires_grad=True))   
        
            for n1 in G.speciesNames:
                if (n,n1) in G.edges:
                    r = 0.5+rand*(np.random.rand()-0.5)
                    G.add_edge(nNewNode-2, n1,
                               w=(r*G.edges[(n,n1)]['w'].clone().detach()).requires_grad_(True))
                    G.add_edge(nNewNode-1, n1, 
                               w=((1-r)*G.edges[(n,n1)]['w'].clone().detach()).requires_grad_(True))
                    G.remove_edge(n,n1)
    G.NinternalNodes = len(G.nodes)-G.Nspecies-1
    assert G.NinternalNodes == nNewNode
    return G


def testSpeciesWeightSumTo1(G):
    '''Test that for each species, the weights on the incoming edges sum to 1 '''
    for node in G.nodes():
        if node in G.speciesNames:
            s = 0.
            for n in G.predecessors(node):
                s = s+G.edges[(n, node)]['w']
            assert torch.allclose(s, torch.tensor(1.))
                    

def testSplit():
    speciesData, speciesNames = getSpecies(Nspecies=3, case='basic')    
    G = getGraph(speciesData, speciesNames, verbose=False)
    plot(G)
    obj = getObj(G)
    print('Obj:', obj)
    obj.backward() 
    print('----f')
    serialize(G, grad=False, printout=True)
    print('----grad')
    serialize(G, grad=True, printout=True)
    zeroGrad(G)
    testSpeciesWeightSumTo1(G)
    
    split(G, rand=0.8)  
    
    print('********************************')
    print(G.edges[(1,'Alpha')])
    print(G.edges[(0,1)])
    
    plot(G)  
    obj = getObj(G)
    print('Obj:', obj)
    obj.backward() 
    print('----f')
    print(serialize(G, grad=False, printout=False))
    serialize(G, grad=False, printout=True)
    print('----grad')
    #print serialize(G, grad=True, printout=False)
    serialize(G, grad=True, printout=True)
    split(G)
    testSpeciesWeightSumTo1(G)
    assert G.NinternalNodes == 7
    plot(G)
    
     
def relaxClose(G, 
               reweight=0.499, 
               splitCondTime=0.3, 
               splitCondWeight=0.5, 
               relabel=True
               ): #TODO: may be buggy
    '''Change network structure near two neighboring nodes having close t values 
    (expected to help overcome potential barier during optimization)
    
    Return: number of splits (= number of added nodes)'''

    for node in G.nodes:
        if node == 'out' or node in G.speciesNames:
            continue
        G.nodes[node]['successorsS'] = list(G.successors(node)) 
        G.nodes[node]['predecessors'] = list(G.predecessors(node))   
    
    # pairs of neighboring nodes where we will put additional nodes
    toSplit = []
    
    internalNodes = [n for n in G.nodes if not (n in G.speciesNames or n == 'out')]
    indSorted = np.argsort([-float(G.nodes[n]['t'].data) for n in internalNodes])
    nodesSorted = [internalNodes[ind] for ind in indSorted]    
    
    for n0 in nodesSorted[::-1]:
        for n1 in G.nodes[n0]['successorsS']:
            if n1 in G.speciesNames:
                continue
            if float(G.edges[(n0,n1)]['w'].data) < splitCondWeight:
                continue 
    
            dt = G.nodes[n1]['t']-G.nodes[n0]['t'] 
            
            assert dt > 0
            L0 = [float((G.nodes[n2]['t']-G.nodes[n0]['t']).data) 
                          for n2 in G.nodes[n0]['successorsS'] if not n2 == n1]
            L1 = [float((G.nodes[n2]['t']-G.nodes[n0]['t']).data) 
                          for n2 in G.nodes[n1]['successorsS']]
            if len(L0) == 0 or len(L1) == 0:
                continue 
                          
            dt1 = np.min(L0+L1)
            assert dt1 >= 0 
            if dt > splitCondTime*dt1:
                continue
            
            toSplit.append((n0,n1))  
    
    if len(toSplit) == 0:
        return 0
    else:    
        # make just one split (TODO: multiple splits) 
        (n0,n1) = toSplit[0]
                    
        G.add_node(G.NinternalNodes,
               t=(G.nodes[n1]['t'].clone().detach()).requires_grad_(True),
               ypos=G.nodes[n0]['ypos'],
               level=G.nodes[n0]['level']+1,
               width=G.nodes[n1]['width'],
               snp=torch.tensor(np.ones((G.batchsize,2)), requires_grad=False))
            
        G.add_edge(n0, G.NinternalNodes,
                   w=torch.tensor(1., requires_grad=True))                     

        for n2 in G.nodes[n0]['successorsS']:            
            if n2 == n1:
                continue
            
            assert G.nodes[G.NinternalNodes]['t'] < G.nodes[n2]['t']
            if (n1,n2) in G.edges:                
                G.add_edge(G.NinternalNodes, n2,
                       w=(G.edges[(n0,n2)]['w'].clone().detach()).requires_grad_(True)) 
                G.remove_edge(n0, n2) 
        
            else:     
                G.add_edge(G.NinternalNodes, n2,
                       w=((1-reweight)*G.edges[(n0,n2)]['w'].clone().detach()).requires_grad_(True)) 
                G.add_edge(n1, n2,
                       w=(reweight*G.edges[(n0,n2)]['w'].clone().detach()).requires_grad_(True)) 
                G.remove_edge(n0, n2)                 
                       
        for n2 in G.nodes[n1]['successorsS']:
            assert G.nodes[G.NinternalNodes]['t'] < G.nodes[n2]['t']
            if (G.NinternalNodes, n2) in G.edges:
                continue
            G.add_edge(G.NinternalNodes, n2,
                   w=(reweight*G.edges[(n1,n2)]['w'].clone().detach()).requires_grad_(True))     
            G.edges[(n1,n2)]['w'] = torch.tensor((1.-reweight)*float(G.edges[(n1,n2)]['w'].data), requires_grad=True)  
            
        L = [float(G.nodes[n3]['t'].data) for n3 in G.nodes[n0]['predecessors'] if not n3 == 'out']
        if len(L) == 0:
            dt = 0.2*float(G.nodes[n0]['t'].data)
        else:
            dt = 0.2*(np.max(L)-float(G.nodes[n0]['t'].data))
        assert dt < 0
        G.nodes[n0]['t'] = torch.tensor(float(G.nodes[n0]['t'].data)+dt, requires_grad=True)

        G.NinternalNodes += 1
        assert G.NinternalNodes == len(G.nodes)-G.Nspecies-1
    
        if relabel:
            internalNodes = [n for n in G.nodes if not (n in G.speciesNames or n == 'out')]
            indSorted = np.argsort([-float(G.nodes[n]['t'].data) for n in internalNodes])[::-1]
            nodesSorted = [internalNodes[ind] for ind in indSorted] 
            D = dict([(node, -k-1) for k, node in enumerate(nodesSorted)])
            D1 = dict([(-k-1, k) for k in range(len(nodesSorted))])
            nx.relabel_nodes(G, D, copy=False)  
            nx.relabel_nodes(G, D1, copy=False) 
            
        return 1 
        
       

def strip(G, 
          threshold=1., # remove the edge if its weight is smaller than the max incoming edge weight times this threshold 
          onlySpecies=False # only strip edges going into species
          ):
    '''Remove edges with low weights, redistributing the weights to other edges'''
    
    if onlySpecies:
        for species in G.speciesNames: 
            maxW = np.max([float(G.edges[(n, species)]['w']) for n in G.predecessors(species)])
            extraW = 0
            neighbors = list(G.predecessors(species))
            for n in neighbors:
                if float(G.edges[(species, n)]['w']) < threshold*maxW:
                    extraW += float(G.edges[(species, n)]['w'])
                    G.remove_edge(n,species)
            
            for n in G.predecessors(species):
                G.edges[(n,species)]['w'] = (1./(1.-extraW)*G.edges[(n,species)]['w'].clone().detach()).requires_grad_(True)
    else:
        for node in G.nodes:
            if node == 'out':
                continue 
            if node == 0:
                assert 'out' in G.predecessors(node)
                continue
            predecessorL = [n for n in G.predecessors(node)]
            maxWind = np.argmax([float(G.edges[(n,node)]['w']) for n in predecessorL])
            maxW = float(G.edges[(predecessorL[maxWind], node)]['w'])
            extraW = 0
            for n in predecessorL:
                if (float(G.edges[(n,node)]['w']) < threshold*maxW or 
                    (float(G.edges[(n,node)]['w']) == threshold*maxW and n != predecessorL[maxWind])):
                    extraW += float(G.edges[(n,node)]['w'])
                    G.remove_edge(n,node)
            
            for n in predecessorL:
                if (n,node) in G.edges:
                    G.edges[(n,node)]['w'] = (1./(1.-extraW)*G.edges[(n,node)]['w'].clone().detach()).requires_grad_(True)        



    
def testStrip():
    speciesData, speciesNames = getSpecies(Nspecies=5, case='basic')    
    G = getGraph(speciesData, speciesNames, verbose=False) 
    testSpeciesWeightSumTo1(G)
    plot(G)
    split(G, rand=1.)
    testSpeciesWeightSumTo1(G)
    plot(G) 
    strip(G, threshold=0.3)
    testSpeciesWeightSumTo1(G)
    plot(G)
    serialize(G, grad=False, printout=True)
    

def getNonSpeciesLeaves(G):
    '''Return list of nodes that are leaves but not species'''
    L = []
    for node in G.nodes:
        if not (node in G.speciesNames or len(list(G.successors(node))) > 0):
            L.append(node)
    return L
        



def stripNodes(G, 
               removeInternal=True, # remove all redundant, not only leaves
               relabel=True, # relabel nodes after removal
               adjustYpos=False): # adjust vertical positions for better plotting
    '''Remove redundant internal nodes (with no branching, left after edge stripping)'''
    
    internalNodes = [n for n in G.nodes if not (n in G.speciesNames or n == 'out')]
    indSorted = np.argsort([-float(G.nodes[n]['t'].data) for n in internalNodes])
    nodesSorted = [internalNodes[ind] for ind in indSorted]
    
    # remove non-species leaves
    for node in nodesSorted:
        if len(list(G.successors(node))) == 0:
            G.remove_node(node)
    
    # remove non-leaf redundand nodes, create new edges if necessary or add weights to existing edges      
    if removeInternal:
        needSweep = True
        while needSweep:
            nodes = list(G.nodes)
            removed = 0
            for node in nodes:
                successors = list(G.successors(node))
                predecessors = list(G.predecessors(node))
                if len(successors) == 1 and len(predecessors) == 1:
                    assert np.allclose(float(G.edges[(node, successors[0])]['w'].data), 1.) or successors[0] in G.speciesNames
                    w = (float(G.edges[(predecessors[0], node)]['w'].data)*  
                         float(G.edges[(node, successors[0])]['w'].data))                  
                    if (predecessors[0], successors[0]) in G.edges:
                        w0 = float(G.edges[(predecessors[0], successors[0])]['w'].data)
                        G.edges[(predecessors[0], successors[0])]['w'] = torch.tensor(w0+w, requires_grad=True) 
                    else:                                                        
                        G.add_edge(predecessors[0], successors[0], w=torch.tensor(w, requires_grad=True))
                    G.remove_node(node)
                    removed += 1
            if removed == 0:
                needSweep = False
        
                
    if relabel:
        internalNodes = [n for n in G.nodes if not (n in G.speciesNames or n == 'out')]
        indSorted = np.argsort([-float(G.nodes[n]['t'].data) for n in internalNodes])[::-1]
        nodesSorted = [internalNodes[ind] for ind in indSorted] 
        D = dict([(node, -k-1) for k, node in enumerate(nodesSorted)])
        D1 = dict([(-k-1, k) for k in range(len(nodesSorted))])
        nx.relabel_nodes(G, D, copy=False)  
        nx.relabel_nodes(G, D1, copy=False) 
        
    if adjustYpos:
        internalNodes = [n for n in G.nodes if not (n in G.speciesNames or n == 'out')]
        indSorted = np.argsort([-float(G.nodes[n]['t'].data) for n in internalNodes])[::-1]
        nodesSorted = [internalNodes[ind] for ind in indSorted]        
        for node in nodesSorted:
            successors = [n for n in G.successors(node) if not (n in G.speciesNames)]
            assert len(successors) <= 2
            for m, n in enumerate(successors):
                # m = 0 or 1
                G.nodes[n]['level'] = G.nodes[node]['level']+1
                G.nodes[n]['ypos'] = G.nodes[node]['ypos']+(2*m-1)*2**(-G.nodes[node]['level']-2)  

    G.NinternalNodes = len(G.nodes)-1-G.Nspecies      
           
            
def testStripNode():
    speciesData, speciesNames = getSpecies(Nspecies=5, case='basic')    
    G = getGraph(speciesData, speciesNames, verbose=False) 
    for _ in range(5):
        split(G, rand=1.)
    strip(G, threshold=1.)
    plot(G) 
    stripNodes(G)  
    plot(G) 
    #print G.NinternalNodes 
    optPosPlot(G)
    plot(G)
    
    
def updateBounds(G, 
                 onlyT=False # only update bounds for the time variables, not edge weights  
                 ):
    '''Update upper and lower bounds for tree parameters to be adjusted by GD'''
    
    k = 0
    for n in range(G.NinternalNodes):         
        G.nodes[n]['bounds'] = [-10., -eps]
        
        # impose constraints that successor nodes are always later in time
        for n1 in list(G.predecessors(n))+list(G.successors(n)):
            if n1 in G.speciesNames or n1 == 'out':
                continue
            if G.nodes[n1]['t'] > G.nodes[n]['t']:
                assert n1 in list(G.successors(n))
                G.nodes[n]['bounds'][1] = np.min([G.nodes[n]['bounds'][1], float(G.nodes[n1]['t'].data)])
            else:
                assert n1 in list(G.predecessors(n))
                G.nodes[n]['bounds'][0] = np.max([G.nodes[n]['bounds'][0], float(G.nodes[n1]['t'].data)])        
        k += 1
        if onlyT:
            continue
        
        # impose constraints that edge weights are positive and <=1
        for n1 in G.speciesNames:
            if (n, n1) in G.edges:
                G.edges[(n,n1)]['bounds'] = [eps, 1.]
                k += 1
        for n1 in range(G.NinternalNodes):
            if (n, n1) in G.edges:
                G.edges[(n,n1)]['bounds'] = [eps, 1.]
                k += 1 
                
                
def backupWeights(G, 
                  rprop=True # rescale gradient components to have magnitude ~1
                  ):
    '''Make a copy of gradient components, to avoid spoiling them during GD update'''
    
    k = 0
    for n in range(G.NinternalNodes): 
        G.nodes[n]['t_backup'] = G.nodes[n]['t'].clone().detach().requires_grad_(False) 
        G.nodes[n]['t_grad'] = G.nodes[n]['t'].grad.clone().detach().requires_grad_(False)     
        k += 1
        for n1 in G.speciesNames:
            if (n, n1) in G.edges:
                G.edges[(n,n1)]['w_backup'] = G.edges[(n,n1)]['w'].clone().detach().requires_grad_(False)
                G.edges[(n,n1)]['w_grad'] = G.edges[(n,n1)]['w'].grad.clone().detach().requires_grad_(False)
                k += 1
        for n1 in range(G.NinternalNodes):
            if (n, n1) in G.edges:
                G.edges[(n,n1)]['w_backup'] = G.edges[(n,n1)]['w'].clone().detach().requires_grad_(False)
                G.edges[(n,n1)]['w_grad'] = G.edges[(n,n1)]['w'].grad.clone().detach().requires_grad_(False)
                k += 1  
                
    if rprop:
        for n in range(G.NinternalNodes):
            if G.nodes[n]['t_grad'] != 0.:
                G.nodes[n]['t_grad'] /= torch.abs(G.nodes[n]['t_grad'])
        for n in G.nodes:
            if n == 'out':
                continue
            G.nodes[n]['predecessors'] = []
            a = 0.
            for n1 in G.predecessors(n):
                if not n1 == 'out':
                    G.nodes[n]['predecessors'].append(n1) 
                    a += G.edges[(n1, n)]['w_grad']  
                    
            if len(G.nodes[n]['predecessors']) == 1:
                G.edges[(n1,n)]['w_grad'] *= 0 
            elif len(G.nodes[n]['predecessors']) >= 2:
                a /= len(G.nodes[n]['predecessors'])
                nPos = 0
                nNeg = 0
                for n1 in G.nodes[n]['predecessors']:
                    G.edges[(n1,n)]['w_grad'] -= a 
                    if G.edges[(n1,n)]['w_grad'] > 0:
                        nPos += 1
                        G.edges[(n1,n)]['w_grad'] /= G.edges[(n1,n)]['w_grad']
                    elif G.edges[(n1,n)]['w_grad'] < 0:
                        nNeg += 1
                        G.edges[(n1,n)]['w_grad'] /= -G.edges[(n1,n)]['w_grad']  
                        
                assert (nPos > 0 and nNeg > 0) or (nPos == 0 and nNeg == 0)  
                if nPos == 0 and nNeg == 0:
                    continue
                for n1 in G.nodes[n]['predecessors']:
                    if G.edges[(n1,n)]['w_grad'] > 0:
                        G.edges[(n1,n)]['w_grad'] /= np.sqrt(float(nNeg)/nPos)
                    elif G.edges[(n1,n)]['w_grad'] < 0:
                        G.edges[(n1,n)]['w_grad'] /= np.sqrt(float(nPos)/nNeg)                                    
                           
                               
def updateWeights(G, 
                  lRate # learning rate
                  ):  
    '''Update tree weights taking into account constraints expressed by 'bounds'   '''
    
    for n in G.nodes:
        if n == 'out':
            continue
        G.nodes[n]['predecessors'] = []
        for n1 in G.predecessors(n):
            if not n1 == 'out':
                G.nodes[n]['predecessors'].append(n1)
                
    k = 0
    for n in range(G.NinternalNodes): 
        if G.nodes[n]['t_grad'] > 0:
            dx = 0.05*(G.nodes[n]['bounds'][1]-float(G.nodes[n]['t_backup'].data))
        else:
            dx = 0.05*(G.nodes[n]['bounds'][0]-float(G.nodes[n]['t_backup'].data))
        G.nodes[n]['t'] = (G.nodes[n]['t_backup']+
                dx*(1.-torch.exp(-lRate*G.nodes[n]['t_grad']/dx))).requires_grad_(True)
        assert G.nodes[n]['t'] <= G.nodes[n]['bounds'][1] and G.nodes[n]['t'] >= G.nodes[n]['bounds'][0]
                                        
        k += 1
        for n1 in G.speciesNames:
            if (n, n1) in G.edges:
                if G.edges[(n,n1)]['w_grad'] > 0:
                    dx = 0.1*(G.edges[(n,n1)]['bounds'][1]-float(G.edges[(n,n1)]['w_backup'].data))
                else:
                    dx = 0.1*(G.edges[(n,n1)]['bounds'][0]-float(G.edges[(n,n1)]['w_backup'].data))
                if np.abs(dx) > 10*eps:
                    G.edges[(n,n1)]['w'] = (G.edges[(n,n1)]['w_backup']+
                        dx*(1.-torch.exp(-lRate*(G.edges[(n,n1)]['w_grad']/dx)))).detach()#.requires_grad_(False)
                else:
                    G.edges[(n,n1)]['w'] = G.edges[(n,n1)]['w_backup'].detach()#.requires_grad_(False)
                assert not torch.isnan(G.edges[(n,n1)]['w'])
                #assert G.edges[(n,n1)]['w'] <= G.edges[(n,n1)]['bounds'][1] and G.edges[(n,n1)]['w'] >= G.edges[(n,n1)]['bounds'][0]
                k += 1
        for n1 in range(G.NinternalNodes):
            if (n, n1) in G.edges:
                if G.edges[(n,n1)]['w_grad'] > 0:
                    dx = G.edges[(n,n1)]['bounds'][1]-float(G.edges[(n,n1)]['w_backup'].data)
                else:
                    dx = G.edges[(n,n1)]['bounds'][0]-float(G.edges[(n,n1)]['w_backup'].data)
                if np.abs(dx) > 10*eps:
                    G.edges[(n,n1)]['w'] = (G.edges[(n,n1)]['w_backup']+
                        dx*(1.-torch.exp(-lRate*(G.edges[(n,n1)]['w_grad']/dx)))).detach()#requires_grad_(False)
                else:
                    G.edges[(n,n1)]['w'] = G.edges[(n,n1)]['w_backup'].detach()#.requires_grad_(False)
                assert not torch.isnan(G.edges[(n,n1)]['w'])
                #assert G.edges[(n,n1)]['w'] <= G.edges[(n,n1)]['bounds'][1] and G.edges[(n,n1)]['w'] >= G.edges[(n,n1)]['bounds'][0]
                k += 1  
                
    for n in G.nodes:
        if n == 'out' or len(G.nodes[n]['predecessors']) == 0:
            continue
        s = 0
        for n1 in G.nodes[n]['predecessors']:
            s += G.edges[(n1,n)]['w']
        for n1 in G.nodes[n]['predecessors']:
            G.edges[(n1,n)]['w'] /= s  
            G.edges[(n1,n)]['w'].requires_grad_(True)    
            
              
def updateGD(G, 
             lRate0=1.,
             verbose=False
             ):
    '''Perform weight updates using a simple line search'''
      
    obj = getObj(G)
    if verbose:
        print('obj:', obj)
    obj.backward() 
    backupWeights(G)
    updateBounds(G, onlyT=False)
    res = []
    lRateL = []
    if verbose:
        print('-----')
        
    # the line search
    for k in range(10):
        lRate = 1.*lRate0*10.**(-0.5*k)
        lRateL.append(lRate)
        updateWeights(G, lRate=lRate)
        obj1 = float(getObj(G).data.numpy())
        if verbose:
            print(k, obj1, obj1-float(obj.data.numpy()))
        res.append(obj1)
        if k > 0 and res[-2]-float(obj.data.numpy()) > 0 and res[-1] < res[-2]:
            break
    k = np.argmax(res)
    assert max(res) >= float(obj.data.numpy())-eps
    if verbose:
        print('=====', k, 10.**(-0.5*k), res[k]-obj.data.numpy())
    updateWeights(G, lRate=lRateL[k])
    if verbose:
        print('new obj:', res[k], float(getObj(G).data.numpy()))
    zeroGrad(G)
    
    
def testUpdateGD():
    np.random.seed(19)
    Gtrue = genRandEvolTree(totalTime=1.0, genomesize=1000, splitRate=2., mutationRate=0.1)
    plot(Gtrue, nonuniformSpecies=True)
    speciesNames = Gtrue.speciesNames
    speciesData = np.array([Gtrue.nodes[species]['genome'] for species in speciesNames])
    
    G = getGraph(speciesData, speciesNames, verbose=False) 
    serialize(G, printout=True)
    
    plot(G)
    lRate0 = 1.
    for s in range(30):
        print('&&&&&&&&&&&&&&&&', s)
        updateGD(G, lRate0, verbose=True)
    
    step0 = 40
    for step in range(30):
        testSpeciesWeightSumTo1(G)
        print('========= STEP', step)  
        print(getObj(G))  
        if step >= step0:    
            plot(G)
        print('Num edges:', G.number_of_edges())
        strip(G, threshold=0.1)
        print('Num edges after stripping:', G.number_of_edges())
        print(getObj(G))
        if step >= step0:
            plot(G)
        print('Num nodes:', G.number_of_nodes())
        stripNodes(G)
        print('Num nodes after stripping:', G.number_of_nodes())
        print(getObj(G))
        if step >= step0:
            plot(G)
        optPosPlot(G)
        if step >= step0:
            plot(G)
        split(G, rand=1e-5)
        print('After splitting:')
        print(getObj(G))
        if step >= step0:
            plot(G)
        nAdded = 1
        while nAdded > 0:
            testSpeciesWeightSumTo1(G)
            nAdded = relaxClose(G)
            testSpeciesWeightSumTo1(G)
            print('After splitting close:')
            print(getObj(G))
            print('nAdded:', nAdded)
            print (getNonSpeciesLeaves(G))
            if step >= step0:
                plot(G)
        
        optPosPlot(G)
        if step >= step0:
            plot(G)
        for s in range(7):
            print('&&&&&&&&&&&&&&&&', s)
            updateGD(G, lRate0)  
            
    serialize(G, printout=True)    
    print(getObj(G))
    plot(G) 
    strip(G, 1.)
    print(getObj(G))
    plot(G)
    stripNodes(G)
    print(getObj(G))
    plot(G)
    optPosPlot(G)
    print(getObj(G))
    plot(G, show=False)
    plot(Gtrue, show=False, nonuniformSpecies=True)
    plt.show()
    
    for step in range(3):
        print('========= STEP', step)  
        print(getObj(G))      
        plot(G)
        print('Num edges:', G.number_of_edges())
        strip(G, threshold=0.1)
        print('Num edges after stripping:', G.number_of_edges())
        print(getObj(G))
        plot(G)
        print('Num nodes:', G.number_of_nodes())
        stripNodes(G)
        print('Num nodes after stripping:', G.number_of_nodes())
        print(getObj(G))
        #plot(G)
        optPosPlot(G)
        plot(G)
        split(G, rand=1e-5)
        print('After splitting:')
        print(getObj(G))
        if step >= step0:
            plot(G)
        nAdded = 1
        while nAdded > 0:
            nAdded = relaxClose(G)
            print('After splitting close:')
            print(getObj(G))
            print('nAdded:', nAdded)
            if step >= step0:
                plot(G)
        
        optPosPlot(G)
        plot(G)
        for s in range(10):
            print('&&&&&&&&&&&&&&&&', s)
            updateGD(G) 
    
    print(getObj(G))
    plot(G) 
    strip(G, 1.)
    print(getObj(G))
    plot(G)
    stripNodes(G)
    print(getObj(G))
    plot(G)
    optPosPlot(G)
    print(getObj(G))
    plot(G, show=False)
    plot(Gtrue, show=False, nonuniformSpecies=True)
    plt.show()      


if __name__ == "__main__":
    testUpdateGD()
   
    