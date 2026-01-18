import networkx as nx
def test_random_cograph():
    n = 3
    G = nx.random_cograph(n)
    assert len(G) == 2 ** n
    if nx.is_connected(G):
        assert nx.diameter(G) <= 2
    else:
        components = nx.connected_components(G)
        for component in components:
            assert nx.diameter(G.subgraph(component)) <= 2