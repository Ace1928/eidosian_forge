from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_radius_weight_None(self):
    assert pytest.approx(nx.radius(self.G, weight=None)) == 2