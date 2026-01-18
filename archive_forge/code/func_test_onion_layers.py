import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_onion_layers(self):
    layers = nx.onion_layers(self.G)
    nodes_by_layer = [sorted((n for n in layers if layers[n] == val)) for val in range(1, 7)]
    assert nodes_equal(nodes_by_layer[0], [21])
    assert nodes_equal(nodes_by_layer[1], [17, 18, 19, 20])
    assert nodes_equal(nodes_by_layer[2], [10, 12, 13, 14, 15, 16])
    assert nodes_equal(nodes_by_layer[3], [9, 11])
    assert nodes_equal(nodes_by_layer[4], [1, 2, 4, 5, 6, 8])
    assert nodes_equal(nodes_by_layer[5], [3, 7])