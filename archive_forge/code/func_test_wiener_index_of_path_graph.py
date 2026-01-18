import networkx as nx
def test_wiener_index_of_path_graph():
    n = 9
    G = nx.path_graph(n)
    expected = 2 * sum((i * (n - i) for i in range(1, n // 2 + 1)))
    actual = nx.wiener_index(G)
    assert expected == actual