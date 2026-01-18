from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_this_one_specific_tree(self):
    """Test the tree pictured at the bottom of [West01]_, p. 78."""
    g = nx.Graph({'a': ['b'], 'b': ['a', 'x'], 'x': ['b', 'y'], 'y': ['x', 'z'], 'z': ['y', 0, 1, 2, 3, 4], 0: ['z'], 1: ['z'], 2: ['z'], 3: ['z'], 4: ['z']})
    b = self.barycenter_as_subgraph(g, attr='barycentricity')
    assert list(b) == ['z']
    assert not b.edges
    expected_barycentricity = {0: 23, 1: 23, 2: 23, 3: 23, 4: 23, 'a': 35, 'b': 27, 'x': 21, 'y': 17, 'z': 15}
    for node, barycentricity in expected_barycentricity.items():
        assert g.nodes[node]['barycentricity'] == barycentricity
    for edge in g.edges:
        g.edges[edge]['weight'] = 2
    b = self.barycenter_as_subgraph(g, weight='weight', attr='barycentricity2')
    assert list(b) == ['z']
    assert not b.edges
    for node, barycentricity in expected_barycentricity.items():
        assert g.nodes[node]['barycentricity2'] == barycentricity * 2