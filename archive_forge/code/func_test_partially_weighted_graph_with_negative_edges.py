import pytest
import networkx as nx
from networkx.utils import pairwise
def test_partially_weighted_graph_with_negative_edges(self):
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 0)])
    G[1][0]['weight'] = -2
    G[0][1]['weight'] = 3
    G[1][2]['weight'] = -4
    H = G.copy()
    H[2][0]['weight'] = 1
    I = G.copy()
    I[2][0]['weight'] = 8
    assert nx.johnson(G) == nx.johnson(H)
    assert nx.johnson(G) != nx.johnson(I)