import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_successful_face_traversal(self):
    embedding = nx.PlanarEmbedding()
    embedding.add_half_edge_first(1, 2)
    embedding.add_half_edge_first(2, 1)
    face = embedding.traverse_face(1, 2)
    assert face == [1, 2]