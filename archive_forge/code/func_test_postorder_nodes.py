import networkx as nx
def test_postorder_nodes(self):
    assert list(nx.dfs_postorder_nodes(self.G, source=0)) == [4, 2, 3, 1, 0]
    assert list(nx.dfs_postorder_nodes(self.D)) == [1, 0, 3, 2]
    assert list(nx.dfs_postorder_nodes(self.D, source=0)) == [1, 0]