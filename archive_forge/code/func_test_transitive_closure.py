from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_transitive_closure(self):
    G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    assert edges_equal(nx.transitive_closure(G).edges(), solution)
    G = nx.DiGraph([(1, 2), (2, 3), (2, 4)])
    solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
    assert edges_equal(nx.transitive_closure(G).edges(), solution)
    G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    solution = [(1, 2), (2, 1), (2, 3), (3, 2), (1, 3), (3, 1)]
    soln = sorted(solution + [(n, n) for n in G])
    assert edges_equal(sorted(nx.transitive_closure(G).edges()), soln)
    G = nx.Graph([(1, 2), (2, 3), (3, 4)])
    solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    assert edges_equal(sorted(nx.transitive_closure(G).edges()), solution)
    G = nx.MultiGraph([(1, 2), (2, 3), (3, 4)])
    solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    assert edges_equal(sorted(nx.transitive_closure(G).edges()), solution)
    G = nx.MultiDiGraph([(1, 2), (2, 3), (3, 4)])
    solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    assert edges_equal(sorted(nx.transitive_closure(G).edges()), solution)
    G = nx.DiGraph([(1, 2, {'a': 3}), (2, 3, {'b': 0}), (3, 4)])
    H = nx.transitive_closure(G)
    for u, v in G.edges():
        assert G.get_edge_data(u, v) == H.get_edge_data(u, v)
    k = 10
    G = nx.DiGraph(((i, i + 1, {'f': 'b', 'weight': i}) for i in range(k)))
    H = nx.transitive_closure(G)
    for u, v in G.edges():
        assert G.get_edge_data(u, v) == H.get_edge_data(u, v)
    G = nx.Graph()
    with pytest.raises(nx.NetworkXError):
        nx.transitive_closure(G, reflexive='wrong input')