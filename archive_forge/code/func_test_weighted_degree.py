import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_weighted_degree(self):
    G = self.Graph()
    G.add_edge(1, 2, weight=2, other=3)
    G.add_edge(2, 3, weight=3, other=4)
    assert sorted((d for n, d in G.degree(weight='weight'))) == [2, 3, 5]
    assert dict(G.degree(weight='weight')) == {1: 2, 2: 5, 3: 3}
    assert G.degree(1, weight='weight') == 2
    assert nodes_equal(G.degree([1], weight='weight'), [(1, 2)])
    assert nodes_equal((d for n, d in G.degree(weight='other')), [3, 7, 4])
    assert dict(G.degree(weight='other')) == {1: 3, 2: 7, 3: 4}
    assert G.degree(1, weight='other') == 3
    assert edges_equal(G.degree([1], weight='other'), [(1, 3)])