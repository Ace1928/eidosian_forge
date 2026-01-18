import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def test_quotient_graph_complete_multipartite():
    """Tests that the quotient graph of the complete *n*-partite graph
    under the "same neighbors" node relation is the complete graph on *n*
    nodes.

    """
    G = nx.complete_multipartite_graph(2, 3, 4)

    def same_neighbors(u, v):
        return u not in G[v] and v not in G[u] and (G[u] == G[v])
    expected = nx.complete_graph(3)
    actual = nx.quotient_graph(G, same_neighbors)
    assert nx.is_isomorphic(expected, actual)