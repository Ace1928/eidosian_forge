import pytest
import networkx as nx
def test_undirected_not_connected(self):
    g = nx.Graph()
    g.add_nodes_from(range(3))
    g.add_edge(0, 1)
    pytest.raises(nx.NetworkXError, nx.average_shortest_path_length, g)