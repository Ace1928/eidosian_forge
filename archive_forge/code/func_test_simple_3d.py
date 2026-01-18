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
def test_simple_3d(self):
    np.random.seed(1234)
    c = np.random.rand(4, 5, 6, 7, 8, 9)
    x = np.linspace(0, 1, 7 + 1)
    y = np.linspace(0, 1, 8 + 1) ** 2
    z = np.linspace(0, 1, 9 + 1) ** 3
    xi = np.random.rand(40)
    yi = np.random.rand(40)
    zi = np.random.rand(40)
    p = NdPPoly(c, (x, y, z))
    for nu in (None, (0, 0, 0), (0, 1, 0), (1, 0, 0), (2, 3, 0), (6, 0, 2)):
        v1 = p((xi, yi, zi), nu=nu)
        v2 = _ppoly3d_eval(c, (x, y, z), xi, yi, zi, nu=nu)
        assert_allclose(v1, v2, err_msg=repr(nu))