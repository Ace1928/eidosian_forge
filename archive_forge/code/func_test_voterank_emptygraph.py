import networkx as nx
def test_voterank_emptygraph(self):
    G = nx.Graph()
    assert [] == nx.voterank(G)