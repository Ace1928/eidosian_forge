import numbers
import pytest
import networkx as nx
from ..generators import (
def test_random_digraph(self):
    n = 10
    m = 20
    G = random_graph(n, m, 0.9, directed=True)
    assert len(G) == 30
    assert nx.is_bipartite(G)
    X, Y = nx.algorithms.bipartite.sets(G)
    assert set(range(n)) == X
    assert set(range(n, n + m)) == Y