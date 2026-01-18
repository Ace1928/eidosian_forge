import math
import networkx as nx
def test_star_broadcast():
    for i in range(4, 12):
        G = nx.star_graph(i)
        b_T, b_C = nx.tree_broadcast_center(G)
        assert b_T == i
        assert b_C == set(G.nodes())
        assert nx.tree_broadcast_time(G) == b_T