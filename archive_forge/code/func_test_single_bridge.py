import pytest
import networkx as nx
def test_single_bridge(self):
    edges = [(1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (1, 3), (1, 4), (2, 5), (5, 10), (6, 8)]
    G = nx.Graph(edges)
    assert nx.has_bridges(G)
    assert nx.has_bridges(G, root=1)