import networkx as nx
def test_number_of_isolates():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_nodes_from([2, 3])
    assert nx.number_of_isolates(G) == 2