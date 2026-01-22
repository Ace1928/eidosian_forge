import pytest
import networkx as nx
from networkx.algorithms.assortativity.correlation import attribute_ac
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
class TestDegreeMixingCorrelation(BaseTestDegreeMixing):

    def test_degree_assortativity_undirected(self):
        r = nx.degree_assortativity_coefficient(self.P4)
        np.testing.assert_almost_equal(r, -1.0 / 2, decimal=4)

    def test_degree_assortativity_node_kwargs(self):
        G = nx.Graph()
        edges = [(0, 1), (0, 3), (1, 2), (1, 3), (1, 4), (5, 9), (9, 0)]
        G.add_edges_from(edges)
        r = nx.degree_assortativity_coefficient(G, nodes=[1, 2, 4])
        np.testing.assert_almost_equal(r, -1.0, decimal=4)

    def test_degree_assortativity_directed(self):
        r = nx.degree_assortativity_coefficient(self.D)
        np.testing.assert_almost_equal(r, -0.57735, decimal=4)

    def test_degree_assortativity_directed2(self):
        """Test degree assortativity for a directed graph where the set of
        in/out degree does not equal the total degree."""
        r = nx.degree_assortativity_coefficient(self.D2)
        np.testing.assert_almost_equal(r, 0.14852, decimal=4)

    def test_degree_assortativity_multigraph(self):
        r = nx.degree_assortativity_coefficient(self.M)
        np.testing.assert_almost_equal(r, -1.0 / 7.0, decimal=4)

    def test_degree_pearson_assortativity_undirected(self):
        r = nx.degree_pearson_correlation_coefficient(self.P4)
        np.testing.assert_almost_equal(r, -1.0 / 2, decimal=4)

    def test_degree_pearson_assortativity_directed(self):
        r = nx.degree_pearson_correlation_coefficient(self.D)
        np.testing.assert_almost_equal(r, -0.57735, decimal=4)

    def test_degree_pearson_assortativity_directed2(self):
        """Test degree assortativity with Pearson for a directed graph where
        the set of in/out degree does not equal the total degree."""
        r = nx.degree_pearson_correlation_coefficient(self.D2)
        np.testing.assert_almost_equal(r, 0.14852, decimal=4)

    def test_degree_pearson_assortativity_multigraph(self):
        r = nx.degree_pearson_correlation_coefficient(self.M)
        np.testing.assert_almost_equal(r, -1.0 / 7.0, decimal=4)

    def test_degree_assortativity_weighted(self):
        r = nx.degree_assortativity_coefficient(self.W, weight='weight')
        np.testing.assert_almost_equal(r, -0.1429, decimal=4)

    def test_degree_assortativity_double_star(self):
        r = nx.degree_assortativity_coefficient(self.DS)
        np.testing.assert_almost_equal(r, -0.9339, decimal=4)