from itertools import combinations
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal
class TestNodeBoundary:
    """Unit tests for the :func:`~networkx.node_boundary` function."""

    def test_null_graph(self):
        """Tests that the null graph has empty node boundaries."""
        null = nx.null_graph()
        assert nx.node_boundary(null, []) == set()
        assert nx.node_boundary(null, [], []) == set()
        assert nx.node_boundary(null, [1, 2, 3]) == set()
        assert nx.node_boundary(null, [1, 2, 3], [4, 5, 6]) == set()
        assert nx.node_boundary(null, [1, 2, 3], [3, 4, 5]) == set()

    def test_path_graph(self):
        P10 = cnlti(nx.path_graph(10), first_label=1)
        assert nx.node_boundary(P10, []) == set()
        assert nx.node_boundary(P10, [], []) == set()
        assert nx.node_boundary(P10, [1, 2, 3]) == {4}
        assert nx.node_boundary(P10, [4, 5, 6]) == {3, 7}
        assert nx.node_boundary(P10, [3, 4, 5, 6, 7]) == {2, 8}
        assert nx.node_boundary(P10, [8, 9, 10]) == {7}
        assert nx.node_boundary(P10, [4, 5, 6], [9, 10]) == set()

    def test_complete_graph(self):
        K10 = cnlti(nx.complete_graph(10), first_label=1)
        assert nx.node_boundary(K10, []) == set()
        assert nx.node_boundary(K10, [], []) == set()
        assert nx.node_boundary(K10, [1, 2, 3]) == {4, 5, 6, 7, 8, 9, 10}
        assert nx.node_boundary(K10, [4, 5, 6]) == {1, 2, 3, 7, 8, 9, 10}
        assert nx.node_boundary(K10, [3, 4, 5, 6, 7]) == {1, 2, 8, 9, 10}
        assert nx.node_boundary(K10, [4, 5, 6], []) == set()
        assert nx.node_boundary(K10, K10) == set()
        assert nx.node_boundary(K10, [1, 2, 3], [3, 4, 5]) == {4, 5}

    def test_petersen(self):
        """Check boundaries in the petersen graph

        cheeger(G,k)=min(|bdy(S)|/|S| for |S|=k, 0<k<=|V(G)|/2)

        """

        def cheeger(G, k):
            return min((len(nx.node_boundary(G, nn)) / k for nn in combinations(G, k)))
        P = nx.petersen_graph()
        assert cheeger(P, 1) == pytest.approx(3.0, abs=0.01)
        assert cheeger(P, 2) == pytest.approx(2.0, abs=0.01)
        assert cheeger(P, 3) == pytest.approx(1.67, abs=0.01)
        assert cheeger(P, 4) == pytest.approx(1.0, abs=0.01)
        assert cheeger(P, 5) == pytest.approx(0.8, abs=0.01)

    def test_directed(self):
        """Tests the node boundary of a directed graph."""
        G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
        S = {0, 1}
        boundary = nx.node_boundary(G, S)
        expected = {2}
        assert boundary == expected

    def test_multigraph(self):
        """Tests the node boundary of a multigraph."""
        G = nx.MultiGraph(list(nx.cycle_graph(5).edges()) * 2)
        S = {0, 1}
        boundary = nx.node_boundary(G, S)
        expected = {2, 4}
        assert boundary == expected

    def test_multidigraph(self):
        """Tests the edge boundary of a multidigraph."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        G = nx.MultiDiGraph(edges * 2)
        S = {0, 1}
        boundary = nx.node_boundary(G, S)
        expected = {2}
        assert boundary == expected