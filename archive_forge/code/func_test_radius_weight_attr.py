from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_radius_weight_attr(self):
    assert pytest.approx(nx.radius(self.G, weight='weight')) == pytest.approx(nx.radius(self.G, weight='cost')) == 0.9 != nx.radius(self.G, weight='high_cost')