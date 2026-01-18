import pytest
import networkx as nx
def test_single_target_shortest_path_length(self):
    pl = nx.single_target_shortest_path_length
    lengths = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
    assert dict(pl(self.cycle, 0)) == lengths
    lengths = {0: 0, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
    assert dict(pl(self.directed_cycle, 0)) == lengths
    target = 8
    with pytest.raises(nx.NodeNotFound, match=f'Target {target} is not in G'):
        nx.single_target_shortest_path_length(self.cycle, target)