import networkx as nx
def test_none_weight_param():
    G = nx.karate_club_graph()
    nx.set_edge_attributes(G, {edge: i * i for i, edge in enumerate(G.edges)}, name='foo')
    part = [{0, 1, 2, 3, 7, 9, 11, 12, 13, 17, 19, 21}, {16, 4, 5, 6, 10}, {23, 25, 27, 28, 24, 31}, {32, 33, 8, 14, 15, 18, 20, 22, 26, 29, 30}]
    partition1 = nx.community.louvain_communities(G, weight=None, seed=2)
    partition2 = nx.community.louvain_communities(G, weight='foo', seed=2)
    partition3 = nx.community.louvain_communities(G, weight='weight', seed=2)
    assert part == partition1
    assert part != partition2
    assert part != partition3
    assert partition2 != partition3