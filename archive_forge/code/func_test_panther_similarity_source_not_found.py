import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_panther_similarity_source_not_found(self):
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4)])
    with pytest.raises(nx.NodeNotFound, match='Source node 10 not in G'):
        nx.panther_similarity(G, source=10)