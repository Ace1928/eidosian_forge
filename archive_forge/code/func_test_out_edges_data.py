import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
def test_out_edges_data(self):
    G = nx.DiGraph([(0, 1, {'data': 0}), (1, 0, {})])
    assert sorted(G.out_edges(data=True)) == [(0, 1, {'data': 0}), (1, 0, {})]
    assert sorted(G.out_edges(0, data=True)) == [(0, 1, {'data': 0})]
    assert sorted(G.out_edges(data='data')) == [(0, 1, 0), (1, 0, None)]
    assert sorted(G.out_edges(0, data='data')) == [(0, 1, 0)]