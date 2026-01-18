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
def test_roots_random(self):
    np.random.seed(1234)
    num = 0
    for extrapolate in (True, False):
        for order in range(0, 20):
            x = np.unique(np.r_[0, 10 * np.random.rand(30), 10])
            c = 2 * np.random.rand(order + 1, len(x) - 1, 2, 3) - 1
            pp = PPoly(c, x)
            for y in [0, np.random.random()]:
                r = pp.solve(y, discontinuity=False, extrapolate=extrapolate)
                for i in range(2):
                    for j in range(3):
                        rr = r[i, j]
                        if rr.size > 0:
                            num += rr.size
                            val = pp(rr, extrapolate=extrapolate)[:, i, j]
                            cmpval = pp(rr, nu=1, extrapolate=extrapolate)[:, i, j]
                            msg = f'({extrapolate!r}) r = {repr(rr)}'
                            assert_allclose((val - y) / cmpval, 0, atol=1e-07, err_msg=msg)
    assert_(num > 100, repr(num))