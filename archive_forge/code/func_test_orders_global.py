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
def test_orders_global(self):
    m, k = (5, 12)
    xi, yi = self._make_random_mk(m, k)
    order = 5
    pp = BPoly.from_derivatives(xi, yi, orders=order)
    for j in range(order // 2 + 1):
        assert_allclose(pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12))
        pp = pp.derivative()
    assert_(not np.allclose(pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12)))
    order = 6
    pp = BPoly.from_derivatives(xi, yi, orders=order)
    for j in range(order // 2):
        assert_allclose(pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12))
        pp = pp.derivative()
    assert_(not np.allclose(pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12)))