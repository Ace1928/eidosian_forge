import pytest
import networkx as nx
def test_smoke_planar_layout_embedding_input(self):
    embedding = nx.PlanarEmbedding()
    embedding.set_data({0: [1, 2], 1: [0, 2], 2: [0, 1]})
    nx.planar_layout(embedding)