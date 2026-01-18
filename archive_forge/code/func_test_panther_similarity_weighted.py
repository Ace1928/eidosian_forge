import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_panther_similarity_weighted(self):
    np.random.seed(42)
    G = nx.Graph()
    G.add_edge('v1', 'v2', w=5)
    G.add_edge('v1', 'v3', w=1)
    G.add_edge('v1', 'v4', w=2)
    G.add_edge('v2', 'v3', w=0.1)
    G.add_edge('v3', 'v5', w=1)
    expected = {'v3': 0.75, 'v4': 0.5, 'v2': 0.5, 'v5': 0.25}
    sim = nx.panther_similarity(G, 'v1', path_length=2, weight='w')
    assert sim == expected