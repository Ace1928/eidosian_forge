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
def _bounds_check_int_nan_fill(self, kind='linear'):
    x = np.arange(10).astype(int)
    y = np.arange(10).astype(int)
    c = interp1d(x, y, kind=kind, fill_value=np.nan, bounds_error=False)
    yi = c(x - 1)
    assert_(np.isnan(yi[0]))
    assert_array_almost_equal(yi, np.r_[np.nan, y[:-1]])