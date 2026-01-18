import networkx as nx
def test_voterank_centrality_1(self):
    G = nx.Graph()
    G.add_edges_from([(7, 8), (7, 5), (7, 9), (5, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 6), (2, 6), (3, 6), (4, 6)])
    assert [0, 7, 6] == nx.voterank(G)