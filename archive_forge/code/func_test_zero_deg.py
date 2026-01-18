from itertools import permutations
import pytest
import networkx as nx
def test_zero_deg(self):
    G = nx.DiGraph()
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(1, 4)
    c = nx.average_degree_connectivity(G)
    assert c == {1: 0, 3: 1}
    c = nx.average_degree_connectivity(G, source='in', target='in')
    assert c == {0: 0, 1: 0}
    c = nx.average_degree_connectivity(G, source='in', target='out')
    assert c == {0: 0, 1: 3}
    c = nx.average_degree_connectivity(G, source='in', target='in+out')
    assert c == {0: 0, 1: 3}
    c = nx.average_degree_connectivity(G, source='out', target='out')
    assert c == {0: 0, 3: 0}
    c = nx.average_degree_connectivity(G, source='out', target='in')
    assert c == {0: 0, 3: 1}
    c = nx.average_degree_connectivity(G, source='out', target='in+out')
    assert c == {0: 0, 3: 1}