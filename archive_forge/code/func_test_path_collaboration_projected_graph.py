import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal, nodes_equal
def test_path_collaboration_projected_graph(self):
    G = nx.path_graph(4)
    P = bipartite.collaboration_weighted_projected_graph(G, [1, 3])
    assert nodes_equal(list(P), [1, 3])
    assert edges_equal(list(P.edges()), [(1, 3)])
    P[1][3]['weight'] = 1
    P = bipartite.collaboration_weighted_projected_graph(G, [0, 2])
    assert nodes_equal(list(P), [0, 2])
    assert edges_equal(list(P.edges()), [(0, 2)])
    P[0][2]['weight'] = 1