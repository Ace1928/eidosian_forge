from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_resistance_distance_all(self):
    rd = nx.resistance_distance(self.G)
    assert type(rd) == dict
    assert round(rd[1][3], 5) == 1