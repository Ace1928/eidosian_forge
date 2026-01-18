import itertools
import pytest
import networkx as nx
def test_num_colors(self):
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3)])
    pytest.raises(nx.NetworkXAlgorithmError, nx.coloring.equitable_color, G, 2)