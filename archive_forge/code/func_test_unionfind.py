import networkx as nx
def test_unionfind():
    x = nx.utils.UnionFind()
    x.union(0, 'a')