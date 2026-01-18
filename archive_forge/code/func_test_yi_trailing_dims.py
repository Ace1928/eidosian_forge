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
def test_yi_trailing_dims(self):
    m, k = (7, 5)
    xi = np.sort(np.random.random(m + 1))
    yi = np.random.random((m + 1, k, 6, 7, 8))
    pp = BPoly.from_derivatives(xi, yi)
    assert_equal(pp.c.shape, (2 * k, m, 6, 7, 8))