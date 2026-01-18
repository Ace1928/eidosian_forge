from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_resistance_distance_only_nodeB(self):
    rd = nx.resistance_distance(self.G, nodeB=1)
    test_data = {}
    test_data[1] = 0
    test_data[2] = 0.75
    test_data[3] = 1
    test_data[4] = 0.75
    assert type(rd) == dict
    assert sorted(rd.keys()) == sorted(test_data.keys())
    for key in rd:
        assert np.isclose(rd[key], test_data[key])