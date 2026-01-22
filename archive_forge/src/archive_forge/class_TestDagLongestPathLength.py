from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
class TestDagLongestPathLength:
    """Unit tests for computing the length of a longest path in a
    directed acyclic graph.

    """

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

    def test_undirected_not_implemented(self):
        G = nx.Graph()
        pytest.raises(nx.NetworkXNotImplemented, nx.dag_longest_path_length, G)

    def test_weighted(self):
        edges = [(1, 2, -5), (2, 3, 1), (3, 4, 1), (4, 5, 0), (3, 5, 4), (1, 6, 2)]
        G = nx.DiGraph()
        G.add_weighted_edges_from(edges)
        assert nx.dag_longest_path_length(G) == 5

    def test_multigraph_unweighted(self):
        edges = [(1, 2), (2, 3), (2, 3), (3, 4), (4, 5), (1, 3), (1, 5), (3, 5)]
        G = nx.MultiDiGraph(edges)
        assert nx.dag_longest_path_length(G) == 4

    def test_multigraph_weighted(self):
        G = nx.MultiDiGraph()
        edges = [(1, 2, 2), (2, 3, 2), (1, 3, 1), (1, 3, 5), (1, 3, 2)]
        G.add_weighted_edges_from(edges)
        assert nx.dag_longest_path_length(G) == 5