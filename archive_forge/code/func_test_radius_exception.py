from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_radius_exception(self):
    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_edge(3, 4)
    pytest.raises(nx.NetworkXError, nx.diameter, G)