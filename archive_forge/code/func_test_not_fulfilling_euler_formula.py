import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_not_fulfilling_euler_formula(self):
    with pytest.raises(nx.NetworkXException):
        embedding = nx.PlanarEmbedding()
        for i in range(5):
            for j in range(5):
                if i != j:
                    embedding.add_half_edge_first(i, j)
        embedding.check_structure()