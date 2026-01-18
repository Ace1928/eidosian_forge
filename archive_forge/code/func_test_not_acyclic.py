from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_not_acyclic(self):
    """Tests that a non-acyclic graph causes an exception."""
    with pytest.raises(nx.HasACycle):
        G = nx.DiGraph(pairwise('abc', cyclic=True))
        nx.dag_to_branching(G)