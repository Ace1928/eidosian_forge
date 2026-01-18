import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def test_undirected_edge_contraction():
    """Tests for edge contraction in an undirected graph."""
    G = nx.cycle_graph(4)
    actual = nx.contracted_edge(G, (0, 1))
    expected = nx.complete_graph(3)
    expected.add_edge(0, 0)
    assert nx.is_isomorphic(actual, expected)