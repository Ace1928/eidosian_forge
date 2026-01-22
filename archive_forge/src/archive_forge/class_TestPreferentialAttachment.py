import math
from functools import partial
import pytest
import networkx as nx
class TestPreferentialAttachment:

    @classmethod
    def setup_class(cls):
        cls.func = staticmethod(nx.preferential_attachment)
        cls.test = partial(_test_func, predict_func=cls.func)

    def test_K5(self):
        G = nx.complete_graph(5)
        self.test(G, [(0, 1)], [(0, 1, 16)])

    def test_P3(self):
        G = nx.path_graph(3)
        self.test(G, [(0, 1)], [(0, 1, 2)])

    def test_S4(self):
        G = nx.star_graph(4)
        self.test(G, [(0, 2)], [(0, 2, 4)])

    def test_notimplemented(self):
        assert pytest.raises(nx.NetworkXNotImplemented, self.func, nx.DiGraph([(0, 1), (1, 2)]), [(0, 2)])
        assert pytest.raises(nx.NetworkXNotImplemented, self.func, nx.MultiGraph([(0, 1), (1, 2)]), [(0, 2)])
        assert pytest.raises(nx.NetworkXNotImplemented, self.func, nx.MultiDiGraph([(0, 1), (1, 2)]), [(0, 2)])

    def test_zero_degrees(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        self.test(G, [(0, 1)], [(0, 1, 0)])

    def test_all_nonexistent_edges(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (2, 3)])
        self.test(G, None, [(0, 3, 2), (1, 2, 2), (1, 3, 1)])