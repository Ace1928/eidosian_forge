import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal, nodes_equal
def test_star_projected_graph(self):
    G = nx.star_graph(3)
    P = bipartite.projected_graph(G, [1, 2, 3])
    assert nodes_equal(list(P), [1, 2, 3])
    assert edges_equal(list(P.edges()), [(1, 2), (1, 3), (2, 3)])
    P = bipartite.weighted_projected_graph(G, [1, 2, 3])
    assert nodes_equal(list(P), [1, 2, 3])
    assert edges_equal(list(P.edges()), [(1, 2), (1, 3), (2, 3)])
    P = bipartite.projected_graph(G, [0])
    assert nodes_equal(list(P), [0])
    assert edges_equal(list(P.edges()), [])