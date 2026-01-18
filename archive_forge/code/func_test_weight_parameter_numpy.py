import pytest
import networkx as nx
def test_weight_parameter_numpy():
    XG4 = nx.Graph()
    XG4.add_edges_from([(0, 1, {'heavy': 2}), (1, 2, {'heavy': 2}), (2, 3, {'heavy': 1}), (3, 4, {'heavy': 1}), (4, 5, {'heavy': 1}), (5, 6, {'heavy': 1}), (6, 7, {'heavy': 1}), (7, 0, {'heavy': 1})])
    dist = nx.floyd_warshall_numpy(XG4, weight='heavy')
    assert dist[0, 2] == 4