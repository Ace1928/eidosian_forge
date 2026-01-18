import networkx as nx
def test_wiener_index_of_complete_graph():
    n = 10
    G = nx.complete_graph(n)
    assert nx.wiener_index(G) == n * (n - 1) / 2