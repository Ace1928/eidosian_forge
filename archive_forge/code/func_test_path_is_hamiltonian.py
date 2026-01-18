from itertools import combinations
import pytest
from networkx import DiGraph
from networkx.algorithms.tournament import (
def test_path_is_hamiltonian():
    G = DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (0, 2)])
    path = hamiltonian_path(G)
    assert len(path) == 4
    assert all((v in G[u] for u, v in zip(path, path[1:])))