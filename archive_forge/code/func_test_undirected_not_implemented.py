from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_undirected_not_implemented(self):
    G = nx.Graph()
    pytest.raises(nx.NetworkXNotImplemented, nx.dag_longest_path_length, G)