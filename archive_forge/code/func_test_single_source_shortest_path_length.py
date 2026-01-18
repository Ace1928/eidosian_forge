import pytest
import networkx as nx
def test_single_source_shortest_path_length(self):
    ans = dict(nx.shortest_path_length(self.cycle, 0))
    assert ans == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
    assert ans == dict(nx.single_source_shortest_path_length(self.cycle, 0))
    ans = dict(nx.shortest_path_length(self.grid, 1))
    assert ans[16] == 6
    ans = dict(nx.shortest_path_length(self.cycle, 0, weight='weight'))
    assert ans == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
    assert ans == dict(nx.single_source_dijkstra_path_length(self.cycle, 0))
    ans = dict(nx.shortest_path_length(self.grid, 1, weight='weight'))
    assert ans[16] == 6
    ans = dict(nx.shortest_path_length(self.cycle, 0, weight='weight', method='dijkstra'))
    assert ans == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
    assert ans == dict(nx.single_source_dijkstra_path_length(self.cycle, 0))
    ans = dict(nx.shortest_path_length(self.cycle, 0, weight='weight', method='bellman-ford'))
    assert ans == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
    assert ans == dict(nx.single_source_bellman_ford_path_length(self.cycle, 0))