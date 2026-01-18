import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_to_directed_as_view(self):
    H = nx.path_graph(2, create_using=self.Graph)
    H2 = H.to_directed(as_view=True)
    assert H is H2._graph
    assert H2.has_edge(0, 1)
    assert H2.has_edge(1, 0) or H.is_directed()
    pytest.raises(nx.NetworkXError, H2.add_node, -1)
    pytest.raises(nx.NetworkXError, H2.add_edge, 1, 2)
    H.add_edge(1, 2)
    assert H2.has_edge(1, 2)
    assert H2.has_edge(2, 1) or H.is_directed()