import networkx as nx
def test_voterank_centrality_2(self):
    G = nx.florentine_families_graph()
    d = nx.voterank(G, 4)
    exact = ['Medici', 'Strozzi', 'Guadagni', 'Castellani']
    assert exact == d