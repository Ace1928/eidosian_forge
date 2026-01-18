from itertools import combinations
import pytest
from networkx import DiGraph
from networkx.algorithms.tournament import (
def test_same_node_is_reachable():
    """Tests that a node is always reachable from it."""
    G = DiGraph((sorted(p) for p in combinations(range(10), 2)))
    assert all((is_reachable(G, v, v) for v in G))