import networkx as nx
def test_partition_iterator():
    G = nx.path_graph(15)
    parts_iter = nx.community.louvain_partitions(G, seed=42)
    first_part = next(parts_iter)
    first_copy = [s.copy() for s in first_part]
    assert first_copy[0] == first_part[0]
    second_part = next(parts_iter)
    assert first_copy[0] == first_part[0]