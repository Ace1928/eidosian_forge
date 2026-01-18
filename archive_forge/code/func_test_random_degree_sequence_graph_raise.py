import pytest
import networkx as nx
def test_random_degree_sequence_graph_raise():
    z = [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4]
    pytest.raises(nx.NetworkXUnfeasible, nx.random_degree_sequence_graph, z)