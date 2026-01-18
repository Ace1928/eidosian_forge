import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_source_target_not_in_graph(self):
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 3, 1)], weight='capacity')
    G.remove_node(0)
    for flow_func in all_funcs:
        pytest.raises(nx.NetworkXError, flow_func, G, 0, 3)
    G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 3, 1)], weight='capacity')
    G.remove_node(3)
    for flow_func in all_funcs:
        pytest.raises(nx.NetworkXError, flow_func, G, 0, 3)