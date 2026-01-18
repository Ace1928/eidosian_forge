import pytest
import networkx as nx
def test_unreachable(self):
    n = 5
    assert n > 1
    G = nx.path_graph(n, create_using=nx.DiGraph())
    assert nx.dominance_frontiers(G, n // 2) == {i: set() for i in range(n // 2, n)}