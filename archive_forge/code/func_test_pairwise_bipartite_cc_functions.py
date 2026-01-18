import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms.bipartite.cluster import cc_dot, cc_max, cc_min
def test_pairwise_bipartite_cc_functions():
    G1 = nx.Graph([(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 5), (1, 6), (1, 7)])
    G2 = nx.Graph([(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (1, 5)])
    G3 = nx.Graph([(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)])
    result = {0: [1 / 3.0, 2 / 3.0, 2 / 5.0], 1: [1 / 2.0, 2 / 3.0, 2 / 3.0], 2: [2 / 8.0, 2 / 5.0, 2 / 5.0]}
    for i, G in enumerate([G1, G2, G3]):
        assert bipartite.is_bipartite(G)
        assert cc_dot(set(G[0]), set(G[1])) == result[i][0]
        assert cc_min(set(G[0]), set(G[1])) == result[i][1]
        assert cc_max(set(G[0]), set(G[1])) == result[i][2]