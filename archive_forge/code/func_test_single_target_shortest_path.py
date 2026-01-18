import pytest
import networkx as nx
def test_single_target_shortest_path(self):
    p = nx.single_target_shortest_path(self.directed_cycle, 0)
    assert p[3] == [3, 4, 5, 6, 0]
    p = nx.single_target_shortest_path(self.cycle, 0)
    assert p[3] == [3, 2, 1, 0]
    p = nx.single_target_shortest_path(self.cycle, 0, cutoff=0)
    assert p == {0: [0]}
    target = 8
    with pytest.raises(nx.NodeNotFound, match=f'Target {target} not in G'):
        nx.single_target_shortest_path(self.cycle, target)