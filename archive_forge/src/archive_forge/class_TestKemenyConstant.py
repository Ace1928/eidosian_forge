from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
class TestKemenyConstant:

    @classmethod
    def setup_class(cls):
        global np
        np = pytest.importorskip('numpy')

    def setup_method(self):
        G = nx.Graph()
        w12 = 2
        w13 = 3
        w23 = 4
        G.add_edge(1, 2, weight=w12)
        G.add_edge(1, 3, weight=w13)
        G.add_edge(2, 3, weight=w23)
        self.G = G

    def test_kemeny_constant_directed(self):
        G = nx.DiGraph()
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(2, 3)
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.kemeny_constant(G)

    def test_kemeny_constant_not_connected(self):
        self.G.add_node(5)
        with pytest.raises(nx.NetworkXError):
            nx.kemeny_constant(self.G)

    def test_kemeny_constant_no_nodes(self):
        G = nx.Graph()
        with pytest.raises(nx.NetworkXError):
            nx.kemeny_constant(G)

    def test_kemeny_constant_negative_weight(self):
        G = nx.Graph()
        w12 = 2
        w13 = 3
        w23 = -10
        G.add_edge(1, 2, weight=w12)
        G.add_edge(1, 3, weight=w13)
        G.add_edge(2, 3, weight=w23)
        with pytest.raises(nx.NetworkXError):
            nx.kemeny_constant(G, weight='weight')

    def test_kemeny_constant(self):
        K = nx.kemeny_constant(self.G, weight='weight')
        w12 = 2
        w13 = 3
        w23 = 4
        test_data = 3 / 2 * (w12 + w13) * (w12 + w23) * (w13 + w23) / (w12 ** 2 * (w13 + w23) + w13 ** 2 * (w12 + w23) + w23 ** 2 * (w12 + w13) + 3 * w12 * w13 * w23)
        assert np.isclose(K, test_data)

    def test_kemeny_constant_no_weight(self):
        K = nx.kemeny_constant(self.G)
        assert np.isclose(K, 4 / 3)

    def test_kemeny_constant_multigraph(self):
        G = nx.MultiGraph()
        w12_1 = 2
        w12_2 = 1
        w13 = 3
        w23 = 4
        G.add_edge(1, 2, weight=w12_1)
        G.add_edge(1, 2, weight=w12_2)
        G.add_edge(1, 3, weight=w13)
        G.add_edge(2, 3, weight=w23)
        K = nx.kemeny_constant(G, weight='weight')
        w12 = w12_1 + w12_2
        test_data = 3 / 2 * (w12 + w13) * (w12 + w23) * (w13 + w23) / (w12 ** 2 * (w13 + w23) + w13 ** 2 * (w12 + w23) + w23 ** 2 * (w12 + w13) + 3 * w12 * w13 * w23)
        assert np.isclose(K, test_data)

    def test_kemeny_constant_weight0(self):
        G = nx.Graph()
        w12 = 0
        w13 = 3
        w23 = 4
        G.add_edge(1, 2, weight=w12)
        G.add_edge(1, 3, weight=w13)
        G.add_edge(2, 3, weight=w23)
        K = nx.kemeny_constant(G, weight='weight')
        test_data = 3 / 2 * (w12 + w13) * (w12 + w23) * (w13 + w23) / (w12 ** 2 * (w13 + w23) + w13 ** 2 * (w12 + w23) + w23 ** 2 * (w12 + w13) + 3 * w12 * w13 * w23)
        assert np.isclose(K, test_data)

    def test_kemeny_constant_selfloop(self):
        G = nx.Graph()
        w11 = 1
        w12 = 2
        w13 = 3
        w23 = 4
        G.add_edge(1, 1, weight=w11)
        G.add_edge(1, 2, weight=w12)
        G.add_edge(1, 3, weight=w13)
        G.add_edge(2, 3, weight=w23)
        K = nx.kemeny_constant(G, weight='weight')
        test_data = (2 * w11 + 3 * w12 + 3 * w13) * (w12 + w23) * (w13 + w23) / ((w12 * w13 + w12 * w23 + w13 * w23) * (w11 + 2 * w12 + 2 * w13 + 2 * w23))
        assert np.isclose(K, test_data)

    def test_kemeny_constant_complete_bipartite_graph(self):
        n1 = 5
        n2 = 4
        G = nx.complete_bipartite_graph(n1, n2)
        K = nx.kemeny_constant(G)
        assert np.isclose(K, n1 + n2 - 3 / 2)

    def test_kemeny_constant_path_graph(self):
        n = 10
        G = nx.path_graph(n)
        K = nx.kemeny_constant(G)
        assert np.isclose(K, n ** 2 / 3 - 2 * n / 3 + 1 / 2)