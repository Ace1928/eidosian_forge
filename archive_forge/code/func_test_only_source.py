import pytest
import networkx as nx
from networkx.algorithms import approximation as approx
def test_only_source():
    G = nx.complete_graph(5)
    pytest.raises(nx.NetworkXError, approx.node_connectivity, G, s=0)