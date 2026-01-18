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
def test_simple_4d(self):
    np.random.seed(1234)
    c = np.random.rand(4, 5, 6, 7, 8, 9, 10, 11)
    x = np.linspace(0, 1, 8 + 1)
    y = np.linspace(0, 1, 9 + 1) ** 2
    z = np.linspace(0, 1, 10 + 1) ** 3
    u = np.linspace(0, 1, 11 + 1) ** 4
    xi = np.random.rand(20)
    yi = np.random.rand(20)
    zi = np.random.rand(20)
    ui = np.random.rand(20)
    p = NdPPoly(c, (x, y, z, u))
    v1 = p((xi, yi, zi, ui))
    v2 = _ppoly4d_eval(c, (x, y, z, u), xi, yi, zi, ui)
    assert_allclose(v1, v2)