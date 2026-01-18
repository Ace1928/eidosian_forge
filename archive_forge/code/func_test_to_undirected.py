from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_multigraph import BaseMultiGraphTester
from .test_multigraph import TestEdgeSubgraph as _TestMultiGraphEdgeSubgraph
from .test_multigraph import TestMultiGraph as _TestMultiGraph
def test_to_undirected(self):
    G = self.K3
    self.add_attributes(G)
    H = nx.MultiGraph(G)
    try:
        assert edges_equal(H.edges(), [(0, 1), (1, 2), (2, 0)])
    except AssertionError:
        assert edges_equal(H.edges(), [(0, 1), (1, 2), (1, 2), (2, 0)])
    H = G.to_undirected()
    self.is_deep(H, G)