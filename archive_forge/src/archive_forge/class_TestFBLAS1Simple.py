import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
class TestFBLAS1Simple:

    def test_axpy(self):
        for p in 'sd':
            f = getattr(fblas, p + 'axpy', None)
            if f is None:
                continue
            assert_array_almost_equal(f([1, 2, 3], [2, -1, 3], a=5), [7, 9, 18])
        for p in 'cz':
            f = getattr(fblas, p + 'axpy', None)
            if f is None:
                continue
            assert_array_almost_equal(f([1, 2j, 3], [2, -1, 3], a=5), [7, 10j - 1, 18])

    def test_copy(self):
        for p in 'sd':
            f = getattr(fblas, p + 'copy', None)
            if f is None:
                continue
            assert_array_almost_equal(f([3, 4, 5], [8] * 3), [3, 4, 5])
        for p in 'cz':
            f = getattr(fblas, p + 'copy', None)
            if f is None:
                continue
            assert_array_almost_equal(f([3, 4j, 5 + 3j], [8] * 3), [3, 4j, 5 + 3j])

    def test_asum(self):
        for p in 'sd':
            f = getattr(fblas, p + 'asum', None)
            if f is None:
                continue
            assert_almost_equal(f([3, -4, 5]), 12)
        for p in ['sc', 'dz']:
            f = getattr(fblas, p + 'asum', None)
            if f is None:
                continue
            assert_almost_equal(f([3j, -4, 3 - 4j]), 14)

    def test_dot(self):
        for p in 'sd':
            f = getattr(fblas, p + 'dot', None)
            if f is None:
                continue
            assert_almost_equal(f([3, -4, 5], [2, 5, 1]), -9)

    def test_complex_dotu(self):
        for p in 'cz':
            f = getattr(fblas, p + 'dotu', None)
            if f is None:
                continue
            assert_almost_equal(f([3j, -4, 3 - 4j], [2, 3, 1]), -9 + 2j)

    def test_complex_dotc(self):
        for p in 'cz':
            f = getattr(fblas, p + 'dotc', None)
            if f is None:
                continue
            assert_almost_equal(f([3j, -4, 3 - 4j], [2, 3j, 1]), 3 - 14j)

    def test_nrm2(self):
        for p in 'sd':
            f = getattr(fblas, p + 'nrm2', None)
            if f is None:
                continue
            assert_almost_equal(f([3, -4, 5]), math.sqrt(50))
        for p in ['c', 'z', 'sc', 'dz']:
            f = getattr(fblas, p + 'nrm2', None)
            if f is None:
                continue
            assert_almost_equal(f([3j, -4, 3 - 4j]), math.sqrt(50))

    def test_scal(self):
        for p in 'sd':
            f = getattr(fblas, p + 'scal', None)
            if f is None:
                continue
            assert_array_almost_equal(f(2, [3, -4, 5]), [6, -8, 10])
        for p in 'cz':
            f = getattr(fblas, p + 'scal', None)
            if f is None:
                continue
            assert_array_almost_equal(f(3j, [3j, -4, 3 - 4j]), [-9, -12j, 12 + 9j])
        for p in ['cs', 'zd']:
            f = getattr(fblas, p + 'scal', None)
            if f is None:
                continue
            assert_array_almost_equal(f(3, [3j, -4, 3 - 4j]), [9j, -12, 9 - 12j])

    def test_swap(self):
        for p in 'sd':
            f = getattr(fblas, p + 'swap', None)
            if f is None:
                continue
            x, y = ([2, 3, 1], [-2, 3, 7])
            x1, y1 = f(x, y)
            assert_array_almost_equal(x1, y)
            assert_array_almost_equal(y1, x)
        for p in 'cz':
            f = getattr(fblas, p + 'swap', None)
            if f is None:
                continue
            x, y = ([2, 3j, 1], [-2, 3, 7 - 3j])
            x1, y1 = f(x, y)
            assert_array_almost_equal(x1, y)
            assert_array_almost_equal(y1, x)

    def test_amax(self):
        for p in 'sd':
            f = getattr(fblas, 'i' + p + 'amax')
            assert_equal(f([-2, 4, 3]), 1)
        for p in 'cz':
            f = getattr(fblas, 'i' + p + 'amax')
            assert_equal(f([-5, 4 + 3j, 6]), 1)