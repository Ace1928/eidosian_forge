import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
class TestBLAS3Syr2k:

    def setup_method(self):
        self.a = np.array([[1.0, 0.0], [0.0, -2.0], [2.0, 3.0]])
        self.b = np.array([[0.0, 1.0], [1.0, 0.0], [0, 1.0]])
        self.t = np.array([[0.0, -1.0, 3.0], [-1.0, 0.0, 0.0], [3.0, 0.0, 6.0]])
        self.tt = np.array([[0.0, 1.0], [1.0, 6]])

    def test_syr2k(self):
        for f in _get_func('syr2k'):
            c = f(a=self.a, b=self.b, alpha=1.0)
            assert_array_almost_equal(np.triu(c), np.triu(self.t))
            c = f(a=self.a, b=self.b, alpha=1.0, lower=1)
            assert_array_almost_equal(np.tril(c), np.tril(self.t))
            c0 = np.ones(self.t.shape)
            c = f(a=self.a, b=self.b, alpha=1.0, beta=1.0, c=c0)
            assert_array_almost_equal(np.triu(c), np.triu(self.t + c0))
            c = f(a=self.a, b=self.b, alpha=1.0, trans=1)
            assert_array_almost_equal(np.triu(c), np.triu(self.tt))

    def test_syr2k_wrong_c(self):
        f = getattr(fblas, 'dsyr2k', None)
        if f is not None:
            assert_raises(Exception, f, **{'a': self.a, 'b': self.b, 'alpha': 1.0, 'c': np.zeros((15, 8))})