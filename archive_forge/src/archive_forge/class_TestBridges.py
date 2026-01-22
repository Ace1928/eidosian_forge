import pytest
import networkx as nx
class TestBridges:
    """Unit tests for the bridge-finding function."""

    def test_single_bridge(self):
        edges = [(1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (1, 3), (1, 4), (2, 5), (5, 10), (6, 8)]
        G = nx.Graph(edges)
        source = 1
        bridges = list(nx.bridges(G, source))
        assert bridges == [(5, 6)]

    def test_barbell_graph(self):
        G = nx.barbell_graph(3, 0)
        source = 0
        bridges = list(nx.bridges(G, source))
        assert bridges == [(2, 3)]

    def test_multiedge_bridge(self):
        edges = [(0, 1), (0, 2), (1, 2), (1, 2), (2, 3), (3, 4), (3, 4)]
        G = nx.MultiGraph(edges)
        assert list(nx.bridges(G)) == [(2, 3)]