import pytest
import networkx as nx
from networkx.generators.degree_seq import havel_hakimi_graph
from networkx.generators.expanders import margulis_gabber_galil_graph
def test_normalized_laplacian(self):
    """Generalized Graph Laplacian"""
    G = np.array([[1.0, -0.408, -0.408, -0.577, 0.0], [-0.408, 1.0, -0.5, 0.0, 0.0], [-0.408, -0.5, 1.0, 0.0, 0.0], [-0.577, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    GL = np.array([[1.0, -0.408, -0.408, -0.577, 0.0], [-0.408, 1.0, -0.5, 0.0, 0.0], [-0.408, -0.5, 1.0, 0.0, 0.0], [-0.577, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    Lsl = np.array([[0.75, -0.2887, -0.2887, -0.3536, 0.0], [-0.2887, 0.6667, -0.3333, 0.0, 0.0], [-0.2887, -0.3333, 0.6667, 0.0, 0.0], [-0.3536, 0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    np.testing.assert_almost_equal(nx.normalized_laplacian_matrix(self.G, nodelist=range(5)).todense(), G, decimal=3)
    np.testing.assert_almost_equal(nx.normalized_laplacian_matrix(self.G).todense(), GL, decimal=3)
    np.testing.assert_almost_equal(nx.normalized_laplacian_matrix(self.MG).todense(), GL, decimal=3)
    np.testing.assert_almost_equal(nx.normalized_laplacian_matrix(self.WG).todense(), GL, decimal=3)
    np.testing.assert_almost_equal(nx.normalized_laplacian_matrix(self.WG, weight='other').todense(), GL, decimal=3)
    np.testing.assert_almost_equal(nx.normalized_laplacian_matrix(self.Gsl).todense(), Lsl, decimal=3)