from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_resistance_distance_not_connected(self):
    with pytest.raises(nx.NetworkXError):
        self.G.add_node(5)
        nx.resistance_distance(self.G, 1, 5)