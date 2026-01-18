import networkx as nx
def test_preorder_nodes(self):
    assert list(nx.dfs_preorder_nodes(self.G, source=0)) == [0, 1, 2, 4, 3]
    assert list(nx.dfs_preorder_nodes(self.D)) == [0, 1, 2, 3]
    assert list(nx.dfs_preorder_nodes(self.D, source=2)) == [2, 3]