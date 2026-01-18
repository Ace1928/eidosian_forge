import networkx as nx
def test_schultz_and_gutman_index_of_disconnected_graph():
    n = 4
    G = nx.Graph()
    G.add_nodes_from(list(range(1, n + 1)))
    expected = float('inf')
    G.add_edge(1, 2)
    G.add_edge(3, 4)
    actual_1 = nx.schultz_index(G)
    actual_2 = nx.gutman_index(G)
    assert expected == actual_1
    assert expected == actual_2