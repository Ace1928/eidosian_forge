import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nonexistent_nodes(self):
    G = nx.complete_graph(5)
    pytest.raises(nx.NetworkXError, nx.common_neighbors, G, 5, 4)
    pytest.raises(nx.NetworkXError, nx.common_neighbors, G, 4, 5)
    pytest.raises(nx.NetworkXError, nx.common_neighbors, G, 5, 6)