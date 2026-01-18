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
def test_raise_degree(self):
    np.random.seed(12345)
    x = [0, 1]
    k, d = (8, 5)
    c = np.random.random((k, 1, 2, 3, 4))
    bp = BPoly(c, x)
    c1 = BPoly._raise_degree(c, d)
    bp1 = BPoly(c1, x)
    xp = np.linspace(0, 1, 11)
    assert_allclose(bp(xp), bp1(xp))