from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_periphery_weight_attr(self):
    periphery = set(nx.periphery(self.G, weight='weight'))
    assert periphery == set(nx.periphery(self.G, weight='cost')) == set(nx.periphery(self.G, weight='high_cost'))
    for v in periphery:
        assert nx.eccentricity(self.G, v, weight='high_cost') != nx.eccentricity(self.G, v, weight='weight') == nx.eccentricity(self.G, v, weight='cost') == nx.diameter(self.G, weight='weight') == nx.diameter(self.G, weight='cost') != nx.diameter(self.G, weight='high_cost')
        assert nx.eccentricity(self.G, v, weight='high_cost') == nx.diameter(self.G, weight='high_cost')