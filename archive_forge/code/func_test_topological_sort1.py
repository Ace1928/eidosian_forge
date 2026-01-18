from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_topological_sort1(self):
    DG = nx.DiGraph([(1, 2), (1, 3), (2, 3)])
    for algorithm in [nx.topological_sort, nx.lexicographical_topological_sort]:
        assert tuple(algorithm(DG)) == (1, 2, 3)
    DG.add_edge(3, 2)
    for algorithm in [nx.topological_sort, nx.lexicographical_topological_sort]:
        pytest.raises(nx.NetworkXUnfeasible, _consume, algorithm(DG))
    DG.remove_edge(2, 3)
    for algorithm in [nx.topological_sort, nx.lexicographical_topological_sort]:
        assert tuple(algorithm(DG)) == (1, 3, 2)
    DG.remove_edge(3, 2)
    assert tuple(nx.topological_sort(DG)) in {(1, 2, 3), (1, 3, 2)}
    assert tuple(nx.lexicographical_topological_sort(DG)) == (1, 2, 3)