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
def _ppoly_eval_1(c, x, xps):
    """Evaluate piecewise polynomial manually"""
    out = np.zeros((len(xps), c.shape[2]))
    for i, xp in enumerate(xps):
        if xp < 0 or xp > 1:
            out[i, :] = np.nan
            continue
        j = np.searchsorted(x, xp) - 1
        d = xp - x[j]
        assert_(x[j] <= xp < x[j + 1])
        r = sum((c[k, j] * d ** (c.shape[0] - k - 1) for k in range(c.shape[0])))
        out[i, :] = r
    return out