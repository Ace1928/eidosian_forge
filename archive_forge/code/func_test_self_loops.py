from itertools import combinations
import pytest
from networkx import DiGraph
from networkx.algorithms.tournament import (
def test_self_loops():
    """A tournament must have no self-loops."""
    G = DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (0, 2)])
    G.add_edge(0, 0)
    assert not is_tournament(G)