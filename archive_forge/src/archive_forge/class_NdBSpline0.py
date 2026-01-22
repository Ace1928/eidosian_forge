import os
import operator
import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
import scipy.linalg as sl
from scipy.interpolate._bsplines import (_not_a_knot, _augknt,
import scipy.interpolate._fitpack_impl as _impl
from scipy._lib._util import AxisError
class NdBSpline0:

    def __init__(self, t, c, k=3):
        """Tensor product spline object.

        c[i1, i2, ..., id] * B(x1, i1) * B(x2, i2) * ... * B(xd, id)

        Parameters
        ----------
        c : ndarray, shape (n1, n2, ..., nd, ...)
            b-spline coefficients
        t : tuple of 1D ndarrays
            knot vectors in directions 1, 2, ... d
            ``len(t[i]) == n[i] + k + 1``
        k : int or length-d tuple of integers
            spline degrees.
        """
        ndim = len(t)
        assert ndim <= len(c.shape)
        try:
            len(k)
        except TypeError:
            k = (k,) * ndim
        self.k = tuple((operator.index(ki) for ki in k))
        self.t = tuple((np.asarray(ti, dtype=float) for ti in t))
        self.c = c

    def __call__(self, x):
        ndim = len(self.t)
        assert len(x) == ndim
        i = ['none'] * ndim
        for d in range(ndim):
            td, xd = (self.t[d], x[d])
            k = self.k[d]
            if xd == td[k]:
                i[d] = k
            else:
                i[d] = np.searchsorted(td, xd) - 1
            assert td[i[d]] <= xd <= td[i[d] + 1]
            assert i[d] >= k and i[d] < len(td) - k
        i = tuple(i)
        result = 0
        iters = [range(i[d] - self.k[d], i[d] + 1) for d in range(ndim)]
        for idx in itertools.product(*iters):
            term = self.c[idx] * np.prod([B(x[d], self.k[d], idx[d], self.t[d]) for d in range(ndim)])
            result += term
        return result