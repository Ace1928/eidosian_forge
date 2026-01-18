import networkx as nx
def test_voterank_centrality_5(self):
    G = nx.MultiDiGraph()
    G.add_edges_from([(0, 1), (0, 1), (1, 2), (2, 5), (2, 5), (5, 6), (5, 6), (2, 4), (4, 3)])
    exact = [2, 0, 5, 4]
    assert exact == nx.voterank(G)