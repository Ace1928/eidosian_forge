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
def test_spline_nans(self):
    x = np.arange(8).astype(float)
    y = x.copy()
    yn = y.copy()
    yn[3] = np.nan
    for kind in ['quadratic', 'cubic']:
        ir = interp1d(x, y, kind=kind)
        irn = interp1d(x, yn, kind=kind)
        for xnew in (6, [1, 6], [[1, 6], [3, 5]]):
            xnew = np.asarray(xnew)
            out, outn = (ir(x), irn(x))
            assert_(np.isnan(outn).all())
            assert_equal(out.shape, outn.shape)