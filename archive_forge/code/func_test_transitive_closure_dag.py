from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_transitive_closure_dag(self):
    G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    transitive_closure = nx.algorithms.dag.transitive_closure_dag
    solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    assert edges_equal(transitive_closure(G).edges(), solution)
    G = nx.DiGraph([(1, 2), (2, 3), (2, 4)])
    solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
    assert edges_equal(transitive_closure(G).edges(), solution)
    G = nx.Graph([(1, 2), (2, 3), (3, 4)])
    pytest.raises(nx.NetworkXNotImplemented, transitive_closure, G)
    G = nx.DiGraph([(1, 2, {'a': 3}), (2, 3, {'b': 0}), (3, 4)])
    H = transitive_closure(G)
    for u, v in G.edges():
        assert G.get_edge_data(u, v) == H.get_edge_data(u, v)
    k = 10
    G = nx.DiGraph(((i, i + 1, {'foo': 'bar', 'weight': i}) for i in range(k)))
    H = transitive_closure(G)
    for u, v in G.edges():
        assert G.get_edge_data(u, v) == H.get_edge_data(u, v)