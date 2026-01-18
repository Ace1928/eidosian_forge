import networkx as nx
def test_wiener_index_of_directed_graph():
    G = nx.complete_graph(3)
    H = nx.DiGraph(G)
    assert 2 * nx.wiener_index(G) == nx.wiener_index(H)