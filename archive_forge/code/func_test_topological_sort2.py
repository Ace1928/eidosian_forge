from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_topological_sort2(self):
    DG = nx.DiGraph({1: [2], 2: [3], 3: [4], 4: [5], 5: [1], 11: [12], 12: [13], 13: [14], 14: [15]})
    pytest.raises(nx.NetworkXUnfeasible, _consume, nx.topological_sort(DG))
    assert not nx.is_directed_acyclic_graph(DG)
    DG.remove_edge(1, 2)
    _consume(nx.topological_sort(DG))
    assert nx.is_directed_acyclic_graph(DG)