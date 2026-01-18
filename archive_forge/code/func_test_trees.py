from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_trees(self):
    """The barycenter of a tree is a single vertex or an edge.

        See [West01]_, p. 78.
        """
    prng = Random(3735928559)
    for i in range(50):
        RT = nx.random_labeled_tree(prng.randint(1, 75), seed=prng)
        b = self.barycenter_as_subgraph(RT)
        if len(b) == 2:
            assert b.size() == 1
        else:
            assert len(b) == 1
            assert b.size() == 0