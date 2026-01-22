import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
class TestMeshgrid:

    def test_simple(self):
        [X, Y] = meshgrid([1, 2, 3], [4, 5, 6, 7])
        assert_array_equal(X, np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]))
        assert_array_equal(Y, np.array([[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]]))

    def test_single_input(self):
        [X] = meshgrid([1, 2, 3, 4])
        assert_array_equal(X, np.array([1, 2, 3, 4]))

    def test_no_input(self):
        args = []
        assert_array_equal([], meshgrid(*args))
        assert_array_equal([], meshgrid(*args, copy=False))

    def test_indexing(self):
        x = [1, 2, 3]
        y = [4, 5, 6, 7]
        [X, Y] = meshgrid(x, y, indexing='ij')
        assert_array_equal(X, np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]))
        assert_array_equal(Y, np.array([[4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7]]))
        z = [8, 9]
        assert_(meshgrid(x, y)[0].shape == (4, 3))
        assert_(meshgrid(x, y, indexing='ij')[0].shape == (3, 4))
        assert_(meshgrid(x, y, z)[0].shape == (4, 3, 2))
        assert_(meshgrid(x, y, z, indexing='ij')[0].shape == (3, 4, 2))
        assert_raises(ValueError, meshgrid, x, y, indexing='notvalid')

    def test_sparse(self):
        [X, Y] = meshgrid([1, 2, 3], [4, 5, 6, 7], sparse=True)
        assert_array_equal(X, np.array([[1, 2, 3]]))
        assert_array_equal(Y, np.array([[4], [5], [6], [7]]))

    def test_invalid_arguments(self):
        assert_raises(TypeError, meshgrid, [1, 2, 3], [4, 5, 6, 7], indices='ij')

    def test_return_type(self):
        x = np.arange(0, 10, dtype=np.float32)
        y = np.arange(10, 20, dtype=np.float64)
        X, Y = np.meshgrid(x, y)
        assert_(X.dtype == x.dtype)
        assert_(Y.dtype == y.dtype)
        X, Y = np.meshgrid(x, y, copy=True)
        assert_(X.dtype == x.dtype)
        assert_(Y.dtype == y.dtype)
        X, Y = np.meshgrid(x, y, sparse=True)
        assert_(X.dtype == x.dtype)
        assert_(Y.dtype == y.dtype)

    def test_writeback(self):
        X = np.array([1.1, 2.2])
        Y = np.array([3.3, 4.4])
        x, y = np.meshgrid(X, Y, sparse=False, copy=True)
        x[0, :] = 0
        assert_equal(x[0, :], 0)
        assert_equal(x[1, :], X)

    def test_nd_shape(self):
        a, b, c, d, e = np.meshgrid(*([0] * i for i in range(1, 6)))
        expected_shape = (2, 1, 3, 4, 5)
        assert_equal(a.shape, expected_shape)
        assert_equal(b.shape, expected_shape)
        assert_equal(c.shape, expected_shape)
        assert_equal(d.shape, expected_shape)
        assert_equal(e.shape, expected_shape)

    def test_nd_values(self):
        a, b, c = np.meshgrid([0], [1, 2], [3, 4, 5])
        assert_equal(a, [[[0, 0, 0]], [[0, 0, 0]]])
        assert_equal(b, [[[1, 1, 1]], [[2, 2, 2]]])
        assert_equal(c, [[[3, 4, 5]], [[3, 4, 5]]])

    def test_nd_indexing(self):
        a, b, c = np.meshgrid([0], [1, 2], [3, 4, 5], indexing='ij')
        assert_equal(a, [[[0, 0, 0], [0, 0, 0]]])
        assert_equal(b, [[[1, 1, 1], [2, 2, 2]]])
        assert_equal(c, [[[3, 4, 5], [3, 4, 5]]])