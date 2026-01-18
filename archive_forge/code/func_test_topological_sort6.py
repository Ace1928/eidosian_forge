from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_topological_sort6(self):
    for algorithm in [nx.topological_sort, nx.lexicographical_topological_sort]:

        def runtime_error():
            DG = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
            first = True
            for x in algorithm(DG):
                if first:
                    first = False
                    DG.add_edge(5 - x, 5)

        def unfeasible_error():
            DG = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
            first = True
            for x in algorithm(DG):
                if first:
                    first = False
                    DG.remove_node(4)

        def runtime_error2():
            DG = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
            first = True
            for x in algorithm(DG):
                if first:
                    first = False
                    DG.remove_node(2)
        pytest.raises(RuntimeError, runtime_error)
        pytest.raises(RuntimeError, runtime_error2)
        pytest.raises(nx.NetworkXUnfeasible, unfeasible_error)