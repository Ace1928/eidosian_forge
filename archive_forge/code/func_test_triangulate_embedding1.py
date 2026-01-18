import math
import pytest
import networkx as nx
from networkx.algorithms.planar_drawing import triangulate_embedding
def test_triangulate_embedding1():
    embedding = nx.PlanarEmbedding()
    embedding.add_node(1)
    expected_embedding = {1: []}
    check_triangulation(embedding, expected_embedding)