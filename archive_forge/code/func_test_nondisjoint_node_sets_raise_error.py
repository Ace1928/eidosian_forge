from itertools import combinations
import pytest
import networkx as nx
def test_nondisjoint_node_sets_raise_error(collider_graph):
    """
    Test that error is raised when node sets aren't disjoint.
    """
    with pytest.raises(nx.NetworkXError):
        nx.is_d_separator(collider_graph, 0, 1, 0)
    with pytest.raises(nx.NetworkXError):
        nx.is_d_separator(collider_graph, 0, 2, 0)
    with pytest.raises(nx.NetworkXError):
        nx.is_d_separator(collider_graph, 0, 0, 1)
    with pytest.raises(nx.NetworkXError):
        nx.is_d_separator(collider_graph, 1, 0, 0)
    with pytest.raises(nx.NetworkXError):
        nx.find_minimal_d_separator(collider_graph, 0, 0)
    with pytest.raises(nx.NetworkXError):
        nx.find_minimal_d_separator(collider_graph, 0, 1, included=0)
    with pytest.raises(nx.NetworkXError):
        nx.find_minimal_d_separator(collider_graph, 1, 0, included=0)
    with pytest.raises(nx.NetworkXError):
        nx.is_minimal_d_separator(collider_graph, 0, 0, set())
    with pytest.raises(nx.NetworkXError):
        nx.is_minimal_d_separator(collider_graph, 0, 1, set(), included=0)
    with pytest.raises(nx.NetworkXError):
        nx.is_minimal_d_separator(collider_graph, 1, 0, set(), included=0)