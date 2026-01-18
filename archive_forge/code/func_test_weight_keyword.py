from itertools import permutations
import pytest
import networkx as nx
def test_weight_keyword(self):
    G = nx.path_graph(4)
    G[1][2]['other'] = 4
    answer = {1: 2.0, 2: 1.8}
    nd = nx.average_degree_connectivity(G, weight='other')
    assert nd == answer
    answer = {1: 2.0, 2: 1.5}
    nd = nx.average_degree_connectivity(G, weight=None)
    assert nd == answer
    D = G.to_directed()
    answer = {2: 2.0, 4: 1.8}
    nd = nx.average_degree_connectivity(D, weight='other')
    assert nd == answer
    answer = {1: 2.0, 2: 1.8}
    D = G.to_directed()
    nd = nx.average_degree_connectivity(D, weight='other', source='in', target='in')
    assert nd == answer
    D = G.to_directed()
    nd = nx.average_degree_connectivity(D, weight='other', source='in', target='in')
    assert nd == answer