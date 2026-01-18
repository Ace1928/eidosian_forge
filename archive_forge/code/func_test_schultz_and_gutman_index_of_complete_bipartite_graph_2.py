import networkx as nx
def test_schultz_and_gutman_index_of_complete_bipartite_graph_2():
    n = 2
    m = 5
    cbg = nx.complete_bipartite_graph(n, m)
    expected_1 = n * m * (n + m) + 2 * n * (n - 1) * m + 2 * m * (m - 1) * n
    actual_1 = nx.schultz_index(cbg)
    expected_2 = n * m * (n * m) + n * (n - 1) * m * m + m * (m - 1) * n * n
    actual_2 = nx.gutman_index(cbg)
    assert expected_1 == actual_1
    assert expected_2 == actual_2