import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_selfloops_attr(self):
    G = self.K3.copy()
    G.add_edge(0, 0)
    G.add_edge(1, 1, weight=2)
    assert edges_equal(nx.selfloop_edges(G, data=True), [(0, 0, {}), (1, 1, {'weight': 2})])
    assert edges_equal(nx.selfloop_edges(G, data='weight'), [(0, 0, None), (1, 1, 2)])