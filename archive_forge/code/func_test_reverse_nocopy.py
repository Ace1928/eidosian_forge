import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
def test_reverse_nocopy(self):
    G = nx.DiGraph([(0, 1), (1, 2)])
    R = G.reverse(copy=False)
    assert sorted(R.edges()) == [(1, 0), (2, 1)]
    with pytest.raises(nx.NetworkXError):
        R.remove_edge(1, 0)