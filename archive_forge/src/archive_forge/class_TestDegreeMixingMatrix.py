import pytest
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
class TestDegreeMixingMatrix(BaseTestDegreeMixing):

    def test_degree_mixing_matrix_undirected(self):
        a_result = np.array([[0, 2], [2, 2]])
        a = nx.degree_mixing_matrix(self.P4, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.degree_mixing_matrix(self.P4)
        np.testing.assert_equal(a, a_result / a_result.sum())

    def test_degree_mixing_matrix_directed(self):
        a_result = np.array([[0, 0, 2], [1, 0, 1], [0, 0, 0]])
        a = nx.degree_mixing_matrix(self.D, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.degree_mixing_matrix(self.D)
        np.testing.assert_equal(a, a_result / a_result.sum())

    def test_degree_mixing_matrix_multigraph(self):
        a_result = np.array([[0, 1, 0], [1, 0, 3], [0, 3, 0]])
        a = nx.degree_mixing_matrix(self.M, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.degree_mixing_matrix(self.M)
        np.testing.assert_equal(a, a_result / a_result.sum())

    def test_degree_mixing_matrix_selfloop(self):
        a_result = np.array([[2]])
        a = nx.degree_mixing_matrix(self.S, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.degree_mixing_matrix(self.S)
        np.testing.assert_equal(a, a_result / a_result.sum())

    def test_degree_mixing_matrix_weighted(self):
        a_result = np.array([[0.0, 1.0], [1.0, 6.0]])
        a = nx.degree_mixing_matrix(self.W, weight='weight', normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.degree_mixing_matrix(self.W, weight='weight')
        np.testing.assert_equal(a, a_result / float(a_result.sum()))

    def test_degree_mixing_matrix_mapping(self):
        a_result = np.array([[6.0, 1.0], [1.0, 0.0]])
        mapping = {0.5: 1, 1.5: 0}
        a = nx.degree_mixing_matrix(self.W, weight='weight', normalized=False, mapping=mapping)
        np.testing.assert_equal(a, a_result)