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
def test_vs_alternative_implementations(self):
    np.random.seed(1234)
    c = np.random.rand(3, 12, 22)
    x = np.sort(np.r_[0, np.random.rand(11), 1])
    p = PPoly(c, x)
    xp = np.r_[0.3, 0.5, 0.33, 0.6]
    expected = _ppoly_eval_1(c, x, xp)
    assert_allclose(p(xp), expected)
    expected = _ppoly_eval_2(c[:, :, 0], x, xp)
    assert_allclose(p(xp)[:, 0], expected)