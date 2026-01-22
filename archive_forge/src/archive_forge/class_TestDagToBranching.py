from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
class TestDagToBranching:
    """Unit tests for the :func:`networkx.dag_to_branching` function."""

    def test_single_root(self):
        """Tests that a directed acyclic graph with a single degree
        zero node produces an arborescence.

        """
        G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
        B = nx.dag_to_branching(G)
        expected = nx.DiGraph([(0, 1), (1, 3), (0, 2), (2, 4)])
        assert nx.is_arborescence(B)
        assert nx.is_isomorphic(B, expected)

    def test_multiple_roots(self):
        """Tests that a directed acyclic graph with multiple degree zero
        nodes creates an arborescence with multiple (weakly) connected
        components.

        """
        G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3), (5, 2)])
        B = nx.dag_to_branching(G)
        expected = nx.DiGraph([(0, 1), (1, 3), (0, 2), (2, 4), (5, 6), (6, 7)])
        assert nx.is_branching(B)
        assert not nx.is_arborescence(B)
        assert nx.is_isomorphic(B, expected)

    def test_already_arborescence(self):
        """Tests that a directed acyclic graph that is already an
        arborescence produces an isomorphic arborescence as output.

        """
        A = nx.balanced_tree(2, 2, create_using=nx.DiGraph())
        B = nx.dag_to_branching(A)
        assert nx.is_isomorphic(A, B)

    def test_already_branching(self):
        """Tests that a directed acyclic graph that is already a
        branching produces an isomorphic branching as output.

        """
        T1 = nx.balanced_tree(2, 2, create_using=nx.DiGraph())
        T2 = nx.balanced_tree(2, 2, create_using=nx.DiGraph())
        G = nx.disjoint_union(T1, T2)
        B = nx.dag_to_branching(G)
        assert nx.is_isomorphic(G, B)

    def test_not_acyclic(self):
        """Tests that a non-acyclic graph causes an exception."""
        with pytest.raises(nx.HasACycle):
            G = nx.DiGraph(pairwise('abc', cyclic=True))
            nx.dag_to_branching(G)

    def test_undirected(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.dag_to_branching(nx.Graph())

    def test_multigraph(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.dag_to_branching(nx.MultiGraph())

    def test_multidigraph(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.dag_to_branching(nx.MultiDiGraph())