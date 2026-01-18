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
def test_simple_1d(self):
    np.random.seed(1234)
    c = np.random.rand(4, 5)
    x = np.linspace(0, 1, 5 + 1)
    xi = np.random.rand(200)
    p = NdPPoly(c, (x,))
    v1 = p((xi,))
    v2 = _ppoly_eval_1(c[:, :, None], x, xi).ravel()
    assert_allclose(v1, v2)