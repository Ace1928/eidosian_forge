from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
class TestDagLongestPath:
    """Unit tests computing the longest path in a directed acyclic graph."""

    def test_empty(self):
        G = nx.DiGraph()
        assert nx.dag_longest_path(G) == []

    def test_unweighted1(self):
        edges = [(1, 2), (2, 3), (2, 4), (3, 5), (5, 6), (3, 7)]
        G = nx.DiGraph(edges)
        assert nx.dag_longest_path(G) == [1, 2, 3, 5, 6]

    def test_unweighted2(self):
        edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 3), (1, 5), (3, 5)]
        G = nx.DiGraph(edges)
        assert nx.dag_longest_path(G) == [1, 2, 3, 4, 5]

    def test_weighted(self):
        G = nx.DiGraph()
        edges = [(1, 2, -5), (2, 3, 1), (3, 4, 1), (4, 5, 0), (3, 5, 4), (1, 6, 2)]
        G.add_weighted_edges_from(edges)
        assert nx.dag_longest_path(G) == [2, 3, 5]

    def test_undirected_not_implemented(self):
        G = nx.Graph()
        pytest.raises(nx.NetworkXNotImplemented, nx.dag_longest_path, G)

    def test_unorderable_nodes(self):
        """Tests that computing the longest path does not depend on
        nodes being orderable.

        For more information, see issue #1989.

        """
        nodes = [object() for n in range(4)]
        G = nx.DiGraph()
        G.add_edge(nodes[0], nodes[1])
        G.add_edge(nodes[0], nodes[2])
        G.add_edge(nodes[2], nodes[3])
        G.add_edge(nodes[1], nodes[3])
        nx.dag_longest_path(G)

    def test_multigraph_unweighted(self):
        edges = [(1, 2), (2, 3), (2, 3), (3, 4), (4, 5), (1, 3), (1, 5), (3, 5)]
        G = nx.MultiDiGraph(edges)
        assert nx.dag_longest_path(G) == [1, 2, 3, 4, 5]

    def test_multigraph_weighted(self):
        G = nx.MultiDiGraph()
        edges = [(1, 2, 2), (2, 3, 2), (1, 3, 1), (1, 3, 5), (1, 3, 2)]
        G.add_weighted_edges_from(edges)
        assert nx.dag_longest_path(G) == [1, 3]

    def test_multigraph_weighted_default_weight(self):
        G = nx.MultiDiGraph([(1, 2), (2, 3)])
        G.add_weighted_edges_from([(1, 3, 1), (1, 3, 5), (1, 3, 2)])
        assert nx.dag_longest_path(G) == [1, 3]
        assert nx.dag_longest_path(G, default_weight=3) == [1, 2, 3]