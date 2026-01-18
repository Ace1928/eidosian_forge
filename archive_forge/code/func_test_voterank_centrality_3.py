import networkx as nx
def test_voterank_centrality_3(self):
    G = nx.gnc_graph(10, seed=7)
    d = nx.voterank(G, 4)
    exact = [3, 6, 8]
    assert exact == d