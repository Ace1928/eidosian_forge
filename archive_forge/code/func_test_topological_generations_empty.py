from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_topological_generations_empty():
    G = nx.DiGraph()
    assert list(nx.topological_generations(G)) == []