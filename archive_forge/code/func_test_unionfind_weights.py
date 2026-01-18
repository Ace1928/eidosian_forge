import networkx as nx
def test_unionfind_weights():
    uf = nx.utils.UnionFind()
    uf.union(1, 4, 7)
    uf.union(2, 5, 8)
    uf.union(3, 6, 9)
    uf.union(1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert uf.weights[uf[1]] == 9