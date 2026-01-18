import collections
import pytest
import networkx as nx
def test_on_eulerian_multigraph(self):
    G = nx.MultiGraph(nx.cycle_graph(3))
    G.add_edge(0, 1)
    H = nx.eulerize(G)
    assert nx.is_eulerian(H)