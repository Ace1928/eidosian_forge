import pytest
import networkx as nx
@pytest.mark.parametrize(('k', 'expected_num_nodes', 'expected_num_edges'), [(2, 10, 10), (4, 10, 20)])
def test_watts_strogatz(k, expected_num_nodes, expected_num_edges):
    G = nx.watts_strogatz_graph(10, k, 0.25, seed=42)
    assert len(G) == expected_num_nodes
    assert G.number_of_edges() == expected_num_edges