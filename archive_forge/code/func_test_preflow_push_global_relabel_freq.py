import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_preflow_push_global_relabel_freq():
    G = nx.DiGraph()
    G.add_edge(1, 2, capacity=1)
    R = preflow_push(G, 1, 2, global_relabel_freq=None)
    assert R.graph['flow_value'] == 1
    pytest.raises(nx.NetworkXError, preflow_push, G, 1, 2, global_relabel_freq=-1)