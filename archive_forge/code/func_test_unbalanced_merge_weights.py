import networkx as nx
def test_unbalanced_merge_weights():
    uf = nx.utils.UnionFind()
    uf.union(1, 2, 3)
    uf.union(4, 5, 6, 7, 8, 9)
    assert uf.weights[uf[1]] == 3
    assert uf.weights[uf[4]] == 6
    largest_root = uf[4]
    uf.union(1, 4)
    assert uf[1] == largest_root
    assert uf.weights[largest_root] == 9