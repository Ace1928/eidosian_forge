import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def test_quotient_graph_edge_relation():
    """Tests for specifying an alternate edge relation for the quotient
    graph.

    """
    G = nx.path_graph(5)

    def identity(u, v):
        return u == v

    def same_parity(b, c):
        return arbitrary_element(b) % 2 == arbitrary_element(c) % 2
    actual = nx.quotient_graph(G, identity, same_parity)
    expected = nx.Graph()
    expected.add_edges_from([(0, 2), (0, 4), (2, 4)])
    expected.add_edge(1, 3)
    assert nx.is_isomorphic(actual, expected)