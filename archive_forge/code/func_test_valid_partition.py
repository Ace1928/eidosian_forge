import networkx as nx
def test_valid_partition():
    G = nx.LFR_benchmark_graph(250, 3, 1.5, 0.009, average_degree=5, min_community=20, seed=10)
    H = G.to_directed()
    partition = nx.community.louvain_communities(G)
    partition2 = nx.community.louvain_communities(H)
    assert nx.community.is_partition(G, partition)
    assert nx.community.is_partition(H, partition2)