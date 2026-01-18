from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_unweighted(self):
    edges = [(1, 2), (2, 3), (2, 4), (3, 5), (5, 6), (5, 7)]
    G = nx.DiGraph(edges)
    assert nx.dag_longest_path_length(G) == 4
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 3), (1, 5), (3, 5)]
    G = nx.DiGraph(edges)
    assert nx.dag_longest_path_length(G) == 4
    G = nx.DiGraph()
    G.add_node(1)
    assert nx.dag_longest_path_length(G) == 0