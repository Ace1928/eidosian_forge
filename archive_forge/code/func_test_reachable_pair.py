from itertools import combinations
import pytest
from networkx import DiGraph
from networkx.algorithms.tournament import (
def test_reachable_pair():
    """Tests for a reachable pair of nodes."""
    G = DiGraph([(0, 1), (1, 2), (2, 0)])
    assert is_reachable(G, 0, 2)