import collections
import pytest
import networkx as nx
def test_on_eulerian(self):
    G = nx.cycle_graph(3)
    H = nx.eulerize(G)
    assert nx.is_isomorphic(G, H)