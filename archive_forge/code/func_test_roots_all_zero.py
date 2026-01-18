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
def test_roots_all_zero(self):
    c = [[0], [0]]
    x = [0, 1]
    p = PPoly(c, x)
    assert_array_equal(p.roots(), [0, np.nan])
    assert_array_equal(p.solve(0), [0, np.nan])
    assert_array_equal(p.solve(1), [])
    c = [[0, 0], [0, 0]]
    x = [0, 1, 2]
    p = PPoly(c, x)
    assert_array_equal(p.roots(), [0, np.nan, 1, np.nan])
    assert_array_equal(p.solve(0), [0, np.nan, 1, np.nan])
    assert_array_equal(p.solve(1), [])