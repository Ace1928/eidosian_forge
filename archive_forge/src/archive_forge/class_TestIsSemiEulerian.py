import collections
import pytest
import networkx as nx
class TestIsSemiEulerian:

    def test_is_semieulerian(self):
        assert nx.is_semieulerian(nx.path_graph(4))
        G = nx.path_graph(6, create_using=nx.DiGraph)
        assert nx.is_semieulerian(G)
        assert not nx.is_semieulerian(nx.complete_graph(5))
        assert not nx.is_semieulerian(nx.complete_graph(7))
        assert not nx.is_semieulerian(nx.hypercube_graph(4))
        assert not nx.is_semieulerian(nx.hypercube_graph(6))