import networkx as nx
def test_size(self):
    G = nx.path_graph(2)
    M = nx.mycielskian(G, 2)
    assert len(M) == 11
    assert M.size() == 20