from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
class TestDistance:

    def setup_method(self):
        G = cnlti(nx.grid_2d_graph(4, 4), first_label=1, ordering='sorted')
        self.G = G

    def test_eccentricity(self):
        assert nx.eccentricity(self.G, 1) == 6
        e = nx.eccentricity(self.G)
        assert e[1] == 6
        sp = dict(nx.shortest_path_length(self.G))
        e = nx.eccentricity(self.G, sp=sp)
        assert e[1] == 6
        e = nx.eccentricity(self.G, v=1)
        assert e == 6
        e = nx.eccentricity(self.G, v=[1, 1])
        assert e[1] == 6
        e = nx.eccentricity(self.G, v=[1, 2])
        assert e[1] == 6
        G = nx.path_graph(1)
        e = nx.eccentricity(G)
        assert e[0] == 0
        e = nx.eccentricity(G, v=0)
        assert e == 0
        pytest.raises(nx.NetworkXError, nx.eccentricity, G, 1)
        G = nx.empty_graph()
        e = nx.eccentricity(G)
        assert e == {}

    def test_diameter(self):
        assert nx.diameter(self.G) == 6

    def test_radius(self):
        assert nx.radius(self.G) == 4

    def test_periphery(self):
        assert set(nx.periphery(self.G)) == {1, 4, 13, 16}

    def test_center(self):
        assert set(nx.center(self.G)) == {6, 7, 10, 11}

    def test_bound_diameter(self):
        assert nx.diameter(self.G, usebounds=True) == 6

    def test_bound_radius(self):
        assert nx.radius(self.G, usebounds=True) == 4

    def test_bound_periphery(self):
        result = {1, 4, 13, 16}
        assert set(nx.periphery(self.G, usebounds=True)) == result

    def test_bound_center(self):
        result = {6, 7, 10, 11}
        assert set(nx.center(self.G, usebounds=True)) == result

    def test_radius_exception(self):
        G = nx.Graph()
        G.add_edge(1, 2)
        G.add_edge(3, 4)
        pytest.raises(nx.NetworkXError, nx.diameter, G)

    def test_eccentricity_infinite(self):
        with pytest.raises(nx.NetworkXError):
            G = nx.Graph([(1, 2), (3, 4)])
            e = nx.eccentricity(G)

    def test_eccentricity_undirected_not_connected(self):
        with pytest.raises(nx.NetworkXError):
            G = nx.Graph([(1, 2), (3, 4)])
            e = nx.eccentricity(G, sp=1)

    def test_eccentricity_directed_weakly_connected(self):
        with pytest.raises(nx.NetworkXError):
            DG = nx.DiGraph([(1, 2), (1, 3)])
            nx.eccentricity(DG)