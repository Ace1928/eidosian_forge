import pytest
import networkx as nx
from networkx.algorithms.community import (
def test_resolution_parameter_impact():
    G = nx.barbell_graph(5, 3)
    gamma = 1
    expected = [frozenset(range(5)), frozenset(range(8, 13)), frozenset(range(5, 8))]
    assert greedy_modularity_communities(G, resolution=gamma) == expected
    assert naive_greedy_modularity_communities(G, resolution=gamma) == expected
    gamma = 2.5
    expected = [{0, 1, 2, 3}, {9, 10, 11, 12}, {5, 6, 7}, {4}, {8}]
    assert greedy_modularity_communities(G, resolution=gamma) == expected
    assert naive_greedy_modularity_communities(G, resolution=gamma) == expected
    gamma = 0.3
    expected = [frozenset(range(8)), frozenset(range(8, 13))]
    assert greedy_modularity_communities(G, resolution=gamma) == expected
    assert naive_greedy_modularity_communities(G, resolution=gamma) == expected