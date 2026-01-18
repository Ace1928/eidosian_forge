import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_unsuccessful_face_traversal(self):
    with pytest.raises(nx.NetworkXException):
        embedding = nx.PlanarEmbedding()
        embedding.add_edge(1, 2, ccw=2, cw=3)
        embedding.add_edge(2, 1, ccw=1, cw=3)
        embedding.traverse_face(1, 2)