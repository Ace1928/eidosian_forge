import pytest
import networkx as nx
def test_shortest_path_length(self):
    assert nx.shortest_path_length(self.cycle, 0, 3) == 3
    assert nx.shortest_path_length(self.grid, 1, 12) == 5
    assert nx.shortest_path_length(self.directed_cycle, 0, 4) == 4
    assert nx.shortest_path_length(self.cycle, 0, 3, weight='weight') == 3
    assert nx.shortest_path_length(self.grid, 1, 12, weight='weight') == 5
    assert nx.shortest_path_length(self.directed_cycle, 0, 4, weight='weight') == 4
    assert nx.shortest_path_length(self.cycle, 0, 3, weight='weight', method='dijkstra') == 3
    assert nx.shortest_path_length(self.cycle, 0, 3, weight='weight', method='bellman-ford') == 3
    pytest.raises(ValueError, nx.shortest_path_length, self.cycle, method='SPAM')
    pytest.raises(nx.NodeNotFound, nx.shortest_path_length, self.cycle, 8)