import pytest
import networkx as nx
class TestTransitivity:

    def test_transitivity(self):
        G = nx.Graph()
        assert nx.transitivity(G) == 0

    def test_path(self):
        G = nx.path_graph(10)
        assert nx.transitivity(G) == 0

    def test_cubical(self):
        G = nx.cubical_graph()
        assert nx.transitivity(G) == 0

    def test_k5(self):
        G = nx.complete_graph(5)
        assert nx.transitivity(G) == 1
        G.remove_edge(1, 2)
        assert nx.transitivity(G) == 0.875