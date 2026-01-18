import networkx as nx
def test_undirected_selfloops():
    G = nx.karate_club_graph()
    expected_partition = nx.community.louvain_communities(G, seed=2, weight=None)
    part = [{0, 1, 2, 3, 7, 9, 11, 12, 13, 17, 19, 21}, {16, 4, 5, 6, 10}, {23, 25, 27, 28, 24, 31}, {32, 33, 8, 14, 15, 18, 20, 22, 26, 29, 30}]
    assert expected_partition == part
    G.add_weighted_edges_from([(i, i, i * 1000) for i in range(9)])
    partition = nx.community.louvain_communities(G, seed=2, weight='weight')
    assert part != partition
    partition = nx.community.louvain_communities(G, seed=2, weight=None)
    assert part == partition