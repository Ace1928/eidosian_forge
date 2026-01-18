import pytest
import networkx as nx
def test_weighted2(self):
    XG4 = nx.Graph()
    XG4.add_weighted_edges_from([[0, 1, 2], [1, 2, 2], [2, 3, 1], [3, 4, 1], [4, 5, 1], [5, 6, 1], [6, 7, 1], [7, 0, 1]])
    path, dist = nx.floyd_warshall_predecessor_and_distance(XG4)
    assert dist[0][2] == 4
    assert path[0][2] == 1