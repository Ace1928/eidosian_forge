import pytest
import networkx as nx
def test_shortest_path_target(self):
    answer = {0: [0, 1], 1: [1], 2: [2, 1]}
    sp = nx.shortest_path(nx.path_graph(3), target=1)
    assert sp == answer
    sp = nx.shortest_path(nx.path_graph(3), target=1, weight='weight')
    assert sp == answer
    sp = nx.shortest_path(nx.path_graph(3), target=1, weight='weight', method='dijkstra')
    assert sp == answer
    sp = nx.shortest_path(nx.path_graph(3), target=1, weight='weight', method='bellman-ford')
    assert sp == answer