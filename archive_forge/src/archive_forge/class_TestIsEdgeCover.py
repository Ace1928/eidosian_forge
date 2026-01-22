import pytest
import networkx as nx
class TestIsEdgeCover:
    """Tests for :func:`networkx.algorithms.is_edge_cover`"""

    def test_empty_graph(self):
        G = nx.Graph()
        assert nx.is_edge_cover(G, set())

    def test_graph_with_loop(self):
        G = nx.Graph()
        G.add_edge(1, 1)
        assert nx.is_edge_cover(G, {(1, 1)})

    def test_graph_single_edge(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        assert nx.is_edge_cover(G, {(0, 0), (1, 1)})
        assert nx.is_edge_cover(G, {(0, 1), (1, 0)})
        assert nx.is_edge_cover(G, {(0, 1)})
        assert not nx.is_edge_cover(G, {(0, 0)})