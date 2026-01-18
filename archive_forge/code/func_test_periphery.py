from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_periphery(self):
    assert set(nx.periphery(self.G)) == {1, 4, 13, 16}