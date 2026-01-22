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
class Test_vectorization_KDTree:

    def setup_method(self):
        self.data = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        self.kdtree = KDTree(self.data)

    def test_single_query(self):
        d, i = self.kdtree.query(np.array([0, 0, 0]))
        assert_(isinstance(d, float))
        assert_(np.issubdtype(i, np.signedinteger))

    def test_vectorized_query(self):
        d, i = self.kdtree.query(np.zeros((2, 4, 3)))
        assert_equal(np.shape(d), (2, 4))
        assert_equal(np.shape(i), (2, 4))

    def test_single_query_multiple_neighbors(self):
        s = 23
        kk = self.kdtree.n + s
        d, i = self.kdtree.query(np.array([0, 0, 0]), k=kk)
        assert_equal(np.shape(d), (kk,))
        assert_equal(np.shape(i), (kk,))
        assert_(np.all(~np.isfinite(d[-s:])))
        assert_(np.all(i[-s:] == self.kdtree.n))

    def test_vectorized_query_multiple_neighbors(self):
        s = 23
        kk = self.kdtree.n + s
        d, i = self.kdtree.query(np.zeros((2, 4, 3)), k=kk)
        assert_equal(np.shape(d), (2, 4, kk))
        assert_equal(np.shape(i), (2, 4, kk))
        assert_(np.all(~np.isfinite(d[:, :, -s:])))
        assert_(np.all(i[:, :, -s:] == self.kdtree.n))

    def test_query_raises_for_k_none(self):
        x = 1.0
        with pytest.raises(ValueError, match='k must be an integer or*'):
            self.kdtree.query(x, k=None)