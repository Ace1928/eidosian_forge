from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
class TestResistanceDistance:

    @classmethod
    def setup_class(cls):
        global np
        np = pytest.importorskip('numpy')

    def setup_method(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=2)
        G.add_edge(2, 3, weight=4)
        G.add_edge(3, 4, weight=1)
        G.add_edge(1, 4, weight=3)
        self.G = G

    def test_resistance_distance_directed_graph(self):
        G = nx.DiGraph()
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.resistance_distance(G)

    def test_resistance_distance_empty(self):
        G = nx.Graph()
        with pytest.raises(nx.NetworkXError):
            nx.resistance_distance(G)

    def test_resistance_distance_not_connected(self):
        with pytest.raises(nx.NetworkXError):
            self.G.add_node(5)
            nx.resistance_distance(self.G, 1, 5)

    def test_resistance_distance_nodeA_not_in_graph(self):
        with pytest.raises(nx.NetworkXError):
            nx.resistance_distance(self.G, 9, 1)

    def test_resistance_distance_nodeB_not_in_graph(self):
        with pytest.raises(nx.NetworkXError):
            nx.resistance_distance(self.G, 1, 9)

    def test_resistance_distance(self):
        rd = nx.resistance_distance(self.G, 1, 3, 'weight', True)
        test_data = 1 / (1 / (2 + 4) + 1 / (1 + 3))
        assert round(rd, 5) == round(test_data, 5)

    def test_resistance_distance_noinv(self):
        rd = nx.resistance_distance(self.G, 1, 3, 'weight', False)
        test_data = 1 / (1 / (1 / 2 + 1 / 4) + 1 / (1 / 1 + 1 / 3))
        assert round(rd, 5) == round(test_data, 5)

    def test_resistance_distance_no_weight(self):
        rd = nx.resistance_distance(self.G, 1, 3)
        assert round(rd, 5) == 1

    def test_resistance_distance_neg_weight(self):
        self.G[2][3]['weight'] = -4
        rd = nx.resistance_distance(self.G, 1, 3, 'weight', True)
        test_data = 1 / (1 / (2 + -4) + 1 / (1 + 3))
        assert round(rd, 5) == round(test_data, 5)

    def test_multigraph(self):
        G = nx.MultiGraph()
        G.add_edge(1, 2, weight=2)
        G.add_edge(2, 3, weight=4)
        G.add_edge(3, 4, weight=1)
        G.add_edge(1, 4, weight=3)
        rd = nx.resistance_distance(G, 1, 3, 'weight', True)
        assert np.isclose(rd, 1 / (1 / (2 + 4) + 1 / (1 + 3)))

    def test_resistance_distance_div0(self):
        with pytest.raises(ZeroDivisionError):
            self.G[1][2]['weight'] = 0
            nx.resistance_distance(self.G, 1, 3, 'weight')

    def test_resistance_distance_same_node(self):
        assert nx.resistance_distance(self.G, 1, 1) == 0

    def test_resistance_distance_only_nodeA(self):
        rd = nx.resistance_distance(self.G, nodeA=1)
        test_data = {}
        test_data[1] = 0
        test_data[2] = 0.75
        test_data[3] = 1
        test_data[4] = 0.75
        assert type(rd) == dict
        assert sorted(rd.keys()) == sorted(test_data.keys())
        for key in rd:
            assert np.isclose(rd[key], test_data[key])

    def test_resistance_distance_only_nodeB(self):
        rd = nx.resistance_distance(self.G, nodeB=1)
        test_data = {}
        test_data[1] = 0
        test_data[2] = 0.75
        test_data[3] = 1
        test_data[4] = 0.75
        assert type(rd) == dict
        assert sorted(rd.keys()) == sorted(test_data.keys())
        for key in rd:
            assert np.isclose(rd[key], test_data[key])

    def test_resistance_distance_all(self):
        rd = nx.resistance_distance(self.G)
        assert type(rd) == dict
        assert round(rd[1][3], 5) == 1