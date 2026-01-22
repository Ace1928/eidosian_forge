import os
from numpy.testing import (assert_equal, assert_array_equal, assert_,
from pytest import raises as assert_raises
import pytest
from platform import python_implementation
import numpy as np
from scipy.spatial import KDTree, Rectangle, distance_matrix, cKDTree
from scipy.spatial._ckdtree import cKDTreeNode
from scipy.spatial import minkowski_distance
import itertools
class sparse_distance_matrix_consistency:

    def distance(self, a, b, p):
        return minkowski_distance(a, b, p)

    def test_consistency_with_neighbors(self):
        M = self.T1.sparse_distance_matrix(self.T2, self.r)
        r = self.T1.query_ball_tree(self.T2, self.r)
        for i, l in enumerate(r):
            for j in l:
                assert_almost_equal(M[i, j], self.distance(self.T1.data[i], self.T2.data[j], self.p), decimal=14)
        for (i, j), d in M.items():
            assert_(j in r[i])

    def test_zero_distance(self):
        self.T1.sparse_distance_matrix(self.T1, self.r)

    def test_consistency(self):
        M1 = self.T1.sparse_distance_matrix(self.T2, self.r)
        expected = distance_matrix(self.T1.data, self.T2.data)
        expected[expected > self.r] = 0
        assert_array_almost_equal(M1.toarray(), expected, decimal=14)

    def test_against_logic_error_regression(self):
        np.random.seed(0)
        too_many = np.array(np.random.randn(18, 2), dtype=int)
        tree = self.kdtree_type(too_many, balanced_tree=False, compact_nodes=False)
        d = tree.sparse_distance_matrix(tree, 3).toarray()
        assert_array_almost_equal(d, d.T, decimal=14)

    def test_ckdtree_return_types(self):
        ref = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                v = self.data1[i, :] - self.data2[j, :]
                ref[i, j] = np.dot(v, v)
        ref = np.sqrt(ref)
        ref[ref > self.r] = 0.0
        dist = np.zeros((self.n, self.n))
        r = self.T1.sparse_distance_matrix(self.T2, self.r, output_type='dict')
        for i, j in r.keys():
            dist[i, j] = r[i, j]
        assert_array_almost_equal(ref, dist, decimal=14)
        dist = np.zeros((self.n, self.n))
        r = self.T1.sparse_distance_matrix(self.T2, self.r, output_type='ndarray')
        for k in range(r.shape[0]):
            i = r['i'][k]
            j = r['j'][k]
            v = r['v'][k]
            dist[i, j] = v
        assert_array_almost_equal(ref, dist, decimal=14)
        r = self.T1.sparse_distance_matrix(self.T2, self.r, output_type='dok_matrix')
        assert_array_almost_equal(ref, r.toarray(), decimal=14)
        r = self.T1.sparse_distance_matrix(self.T2, self.r, output_type='coo_matrix')
        assert_array_almost_equal(ref, r.toarray(), decimal=14)