import networkx as nx
def test_predecessor(self):
    assert nx.dfs_predecessors(self.G, source=0) == {1: 0, 2: 1, 3: 1, 4: 2}
    assert nx.dfs_predecessors(self.D) == {1: 0, 3: 2}