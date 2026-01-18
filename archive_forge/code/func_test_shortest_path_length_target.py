import pytest
import networkx as nx
def test_shortest_path_length_target(self):
    answer = {0: 1, 1: 0, 2: 1}
    sp = dict(nx.shortest_path_length(nx.path_graph(3), target=1))
    assert sp == answer
    sp = nx.shortest_path_length(nx.path_graph(3), target=1, weight='weight')
    assert sp == answer
    sp = nx.shortest_path_length(nx.path_graph(3), target=1, weight='weight', method='dijkstra')
    assert sp == answer
    sp = nx.shortest_path_length(nx.path_graph(3), target=1, weight='weight', method='bellman-ford')
    assert sp == answer