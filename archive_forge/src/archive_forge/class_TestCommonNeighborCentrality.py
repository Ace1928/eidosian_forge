import math
from functools import partial
import pytest
import networkx as nx
class TestCommonNeighborCentrality:

    @classmethod
    def setup_class(cls):
        cls.func = staticmethod(nx.common_neighbor_centrality)
        cls.test = partial(_test_func, predict_func=cls.func)

    def test_K5(self):
        G = nx.complete_graph(5)
        self.test(G, [(0, 1)], [(0, 1, 3.0)], alpha=1)
        self.test(G, [(0, 1)], [(0, 1, 5.0)], alpha=0)

    def test_P3(self):
        G = nx.path_graph(3)
        self.test(G, [(0, 2)], [(0, 2, 1.25)], alpha=0.5)

    def test_S4(self):
        G = nx.star_graph(4)
        self.test(G, [(1, 2)], [(1, 2, 1.75)], alpha=0.5)

    @pytest.mark.parametrize('graph_type', (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
    def test_notimplemented(self, graph_type):
        assert pytest.raises(nx.NetworkXNotImplemented, self.func, graph_type([(0, 1), (1, 2)]), [(0, 2)])

    def test_no_common_neighbor(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        self.test(G, [(0, 1)], [(0, 1, 0)])

    def test_equal_nodes(self):
        G = nx.complete_graph(4)
        assert pytest.raises(nx.NetworkXAlgorithmError, self.test, G, [(0, 0)], [])

    def test_all_nonexistent_edges(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (2, 3)])
        self.test(G, None, [(0, 3, 1.5), (1, 2, 1.5), (1, 3, 2 / 3)], alpha=0.5)