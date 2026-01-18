import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_selfloop_degree(self):
    G = self.Graph()
    G.add_edge(1, 1)
    assert sorted(G.degree()) == [(1, 2)]
    assert dict(G.degree()) == {1: 2}
    assert G.degree(1) == 2
    assert sorted(G.degree([1])) == [(1, 2)]
    assert G.degree(1, weight='weight') == 2