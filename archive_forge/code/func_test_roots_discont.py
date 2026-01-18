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
def test_roots_discont(self):
    c = np.array([[1], [-1]]).T
    x = np.array([0, 0.5, 1])
    pp = PPoly(c, x)
    assert_array_equal(pp.roots(), [0.5])
    assert_array_equal(pp.roots(discontinuity=False), [])
    assert_array_equal(pp.solve(0.5), [0.5])
    assert_array_equal(pp.solve(0.5, discontinuity=False), [])
    assert_array_equal(pp.solve(1.5), [])
    assert_array_equal(pp.solve(1.5, discontinuity=False), [])