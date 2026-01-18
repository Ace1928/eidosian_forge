from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_resistance_distance_directed_graph(self):
    G = nx.DiGraph()
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.resistance_distance(G)