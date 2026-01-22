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
def _nd_check_shape(self, kind='linear'):
    a = [4, 5, 6, 7]
    y = np.arange(np.prod(a)).reshape(*a)
    for n, s in enumerate(a):
        x = np.arange(s)
        z = interp1d(x, y, axis=n, kind=kind)
        assert_array_almost_equal(z(x), y, err_msg=kind)
        x2 = np.arange(2 * 3 * 1).reshape((2, 3, 1)) / 12.0
        b = list(a)
        b[n:n + 1] = [2, 3, 1]
        assert_array_almost_equal(z(x2).shape, b, err_msg=kind)