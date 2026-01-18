from itertools import combinations
import pytest
from networkx import DiGraph
from networkx.algorithms.tournament import (
def test_unreachable_pair():
    """Tests for an unreachable pair of nodes."""
    G = DiGraph([(0, 1), (0, 2), (1, 2)])
    assert not is_reachable(G, 1, 0)