import importlib.resources
import os
import random
import struct
import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_noncomparable_nodes():
    node1 = object()
    node2 = object()
    node3 = object()
    G = nx.path_graph([node1, node2, node3])
    gm = iso.GraphMatcher(G, G)
    assert gm.is_isomorphic()
    assert gm.subgraph_is_monomorphic()
    G = nx.path_graph([node1, node2, node3], create_using=nx.DiGraph)
    H = nx.path_graph([node3, node2, node1], create_using=nx.DiGraph)
    dgm = iso.DiGraphMatcher(G, H)
    assert dgm.is_isomorphic()
    assert gm.subgraph_is_monomorphic()