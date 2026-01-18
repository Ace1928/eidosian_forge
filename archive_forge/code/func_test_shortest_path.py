import pytest
import networkx as nx
def test_shortest_path(self):
    assert nx.shortest_path(self.cycle, 0, 3) == [0, 1, 2, 3]
    assert nx.shortest_path(self.cycle, 0, 4) == [0, 6, 5, 4]
    validate_grid_path(4, 4, 1, 12, nx.shortest_path(self.grid, 1, 12))
    assert nx.shortest_path(self.directed_cycle, 0, 3) == [0, 1, 2, 3]
    assert nx.shortest_path(self.cycle, 0, 3, weight='weight') == [0, 1, 2, 3]
    assert nx.shortest_path(self.cycle, 0, 4, weight='weight') == [0, 6, 5, 4]
    validate_grid_path(4, 4, 1, 12, nx.shortest_path(self.grid, 1, 12, weight='weight'))
    assert nx.shortest_path(self.directed_cycle, 0, 3, weight='weight') == [0, 1, 2, 3]
    assert nx.shortest_path(self.directed_cycle, 0, 3, weight='weight', method='dijkstra') == [0, 1, 2, 3]
    assert nx.shortest_path(self.directed_cycle, 0, 3, weight='weight', method='bellman-ford') == [0, 1, 2, 3]
    assert nx.shortest_path(self.neg_weights, 0, 3, weight='weight', method='bellman-ford') == [0, 2, 3]
    pytest.raises(ValueError, nx.shortest_path, self.cycle, method='SPAM')
    pytest.raises(nx.NodeNotFound, nx.shortest_path, self.cycle, 8)