import pytest
import networkx as nx
from networkx.algorithms.planarity import (
class TestPlanarEmbeddingClass:

    def test_get_data(self):
        embedding = self.get_star_embedding(3)
        data = embedding.get_data()
        data_cmp = {0: [2, 1], 1: [0], 2: [0]}
        assert data == data_cmp

    def test_missing_edge_orientation(self):
        with pytest.raises(nx.NetworkXException):
            embedding = nx.PlanarEmbedding()
            embedding.add_edge(1, 2)
            embedding.add_edge(2, 1)
            embedding.check_structure()

    def test_invalid_edge_orientation(self):
        with pytest.raises(nx.NetworkXException):
            embedding = nx.PlanarEmbedding()
            embedding.add_half_edge_first(1, 2)
            embedding.add_half_edge_first(2, 1)
            embedding.add_edge(1, 3)
            embedding.check_structure()

    def test_missing_half_edge(self):
        with pytest.raises(nx.NetworkXException):
            embedding = nx.PlanarEmbedding()
            embedding.add_half_edge_first(1, 2)
            embedding.check_structure()

    def test_not_fulfilling_euler_formula(self):
        with pytest.raises(nx.NetworkXException):
            embedding = nx.PlanarEmbedding()
            for i in range(5):
                for j in range(5):
                    if i != j:
                        embedding.add_half_edge_first(i, j)
            embedding.check_structure()

    def test_missing_reference(self):
        with pytest.raises(nx.NetworkXException):
            embedding = nx.PlanarEmbedding()
            embedding.add_half_edge_cw(1, 2, 3)

    def test_connect_components(self):
        embedding = nx.PlanarEmbedding()
        embedding.connect_components(1, 2)

    def test_successful_face_traversal(self):
        embedding = nx.PlanarEmbedding()
        embedding.add_half_edge_first(1, 2)
        embedding.add_half_edge_first(2, 1)
        face = embedding.traverse_face(1, 2)
        assert face == [1, 2]

    def test_unsuccessful_face_traversal(self):
        with pytest.raises(nx.NetworkXException):
            embedding = nx.PlanarEmbedding()
            embedding.add_edge(1, 2, ccw=2, cw=3)
            embedding.add_edge(2, 1, ccw=1, cw=3)
            embedding.traverse_face(1, 2)

    @staticmethod
    def get_star_embedding(n):
        embedding = nx.PlanarEmbedding()
        for i in range(1, n):
            embedding.add_half_edge_first(0, i)
            embedding.add_half_edge_first(i, 0)
        return embedding