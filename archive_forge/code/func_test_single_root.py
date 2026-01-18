from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_single_root(self):
    """Tests that a directed acyclic graph with a single degree
        zero node produces an arborescence.

        """
    G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    B = nx.dag_to_branching(G)
    expected = nx.DiGraph([(0, 1), (1, 3), (0, 2), (2, 4)])
    assert nx.is_arborescence(B)
    assert nx.is_isomorphic(B, expected)