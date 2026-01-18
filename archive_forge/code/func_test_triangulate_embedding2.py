import math
import pytest
import networkx as nx
from networkx.algorithms.planar_drawing import triangulate_embedding
def test_triangulate_embedding2():
    embedding = nx.PlanarEmbedding()
    embedding.connect_components(1, 2)
    expected_embedding = {1: [2], 2: [1]}
    check_triangulation(embedding, expected_embedding)