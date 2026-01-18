from numpy.testing import (assert_, assert_equal, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
from numpy import mgrid, pi, sin, ogrid, poly1d, linspace
import numpy as np
from scipy.interpolate import (interp1d, interp2d, lagrange, PPoly, BPoly,
from scipy.special import poch, gamma
from scipy.interpolate import _ppoly
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
from scipy.integrate import nquad
from scipy.special import binom
def test_simple_2d(self):
    np.random.seed(1234)
    c = np.random.rand(4, 5, 6, 7)
    x = np.linspace(0, 1, 6 + 1)
    y = np.linspace(0, 1, 7 + 1) ** 2
    xi = np.random.rand(200)
    yi = np.random.rand(200)
    v1 = np.empty([len(xi), 1], dtype=c.dtype)
    v1.fill(np.nan)
    _ppoly.evaluate_nd(c.reshape(4 * 5, 6 * 7, 1), (x, y), np.array([4, 5], dtype=np.intc), np.c_[xi, yi], np.array([0, 0], dtype=np.intc), 1, v1)
    v1 = v1.ravel()
    v2 = _ppoly2d_eval(c, (x, y), xi, yi)
    assert_allclose(v1, v2)
    p = NdPPoly(c, (x, y))
    for nu in (None, (0, 0), (0, 1), (1, 0), (2, 3), (9, 2)):
        v1 = p(np.c_[xi, yi], nu=nu)
        v2 = _ppoly2d_eval(c, (x, y), xi, yi, nu=nu)
        assert_allclose(v1, v2, err_msg=repr(nu))