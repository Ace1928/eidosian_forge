import pytest
import networkx as nx
def test_weight_parameter(self):
    XG4 = nx.Graph()
    XG4.add_edges_from([(0, 1, {'heavy': 2}), (1, 2, {'heavy': 2}), (2, 3, {'heavy': 1}), (3, 4, {'heavy': 1}), (4, 5, {'heavy': 1}), (5, 6, {'heavy': 1}), (6, 7, {'heavy': 1}), (7, 0, {'heavy': 1})])
    path, dist = nx.floyd_warshall_predecessor_and_distance(XG4, weight='heavy')
    assert dist[0][2] == 4
    assert path[0][2] == 1