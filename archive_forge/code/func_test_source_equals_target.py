import pytest
import networkx as nx
from networkx.algorithms import approximation as approx
def test_source_equals_target():
    G = nx.complete_graph(5)
    pytest.raises(nx.NetworkXError, approx.local_node_connectivity, G, 0, 0)