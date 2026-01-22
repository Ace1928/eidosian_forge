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
def _nd_check_interp(self, kind='linear'):
    interp10 = interp1d(self.x10, self.y10, kind=kind)
    assert_array_almost_equal(interp10(np.array([[3.0, 5.0], [2.0, 7.0]])), np.array([[3.0, 5.0], [2.0, 7.0]]))
    assert_(isinstance(interp10(1.2), np.ndarray))
    assert_equal(interp10(1.2).shape, ())
    interp210 = interp1d(self.x10, self.y210, kind=kind)
    assert_array_almost_equal(interp210(1.0), np.array([1.0, 11.0]))
    assert_array_almost_equal(interp210(np.array([1.0, 2.0])), np.array([[1.0, 2.0], [11.0, 12.0]]))
    interp102 = interp1d(self.x10, self.y102, axis=0, kind=kind)
    assert_array_almost_equal(interp102(1.0), np.array([2.0, 3.0]))
    assert_array_almost_equal(interp102(np.array([1.0, 3.0])), np.array([[2.0, 3.0], [6.0, 7.0]]))
    x_new = np.array([[3.0, 5.0], [2.0, 7.0]])
    assert_array_almost_equal(interp210(x_new), np.array([[[3.0, 5.0], [2.0, 7.0]], [[13.0, 15.0], [12.0, 17.0]]]))
    assert_array_almost_equal(interp102(x_new), np.array([[[6.0, 7.0], [10.0, 11.0]], [[4.0, 5.0], [14.0, 15.0]]]))