import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
def test_pred_succ(self):
    """Test that nodes are added to predecessors and successors.

        For more information, see GitHub issue #2370.

        """
    G = nx.DiGraph()
    G.add_edge(0, 1)
    H = G.edge_subgraph([(0, 1)])
    assert list(H.predecessors(0)) == []
    assert list(H.successors(0)) == [1]
    assert list(H.predecessors(1)) == [0]
    assert list(H.successors(1)) == []