import os
import operator
import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
import scipy.linalg as sl
from scipy.interpolate._bsplines import (_not_a_knot, _augknt,
import scipy.interpolate._fitpack_impl as _impl
from scipy._lib._util import AxisError
def test_string_aliases(self):
    yy = np.sin(self.xx)
    b1 = make_interp_spline(self.xx, yy, k=3, bc_type='natural')
    b2 = make_interp_spline(self.xx, yy, k=3, bc_type=([(2, 0)], [(2, 0)]))
    assert_allclose(b1.c, b2.c, atol=1e-15)
    b1 = make_interp_spline(self.xx, yy, k=3, bc_type=('natural', 'clamped'))
    b2 = make_interp_spline(self.xx, yy, k=3, bc_type=([(2, 0)], [(1, 0)]))
    assert_allclose(b1.c, b2.c, atol=1e-15)
    b1 = make_interp_spline(self.xx, yy, k=2, bc_type=(None, 'clamped'))
    b2 = make_interp_spline(self.xx, yy, k=2, bc_type=(None, [(1, 0.0)]))
    assert_allclose(b1.c, b2.c, atol=1e-15)
    b1 = make_interp_spline(self.xx, yy, k=3, bc_type='not-a-knot')
    b2 = make_interp_spline(self.xx, yy, k=3, bc_type=None)
    assert_allclose(b1.c, b2.c, atol=1e-15)
    with assert_raises(ValueError):
        make_interp_spline(self.xx, yy, k=3, bc_type='typo')
    yy = np.c_[np.sin(self.xx), np.cos(self.xx)]
    der_l = [(1, [0.0, 0.0])]
    der_r = [(2, [0.0, 0.0])]
    b2 = make_interp_spline(self.xx, yy, k=3, bc_type=(der_l, der_r))
    b1 = make_interp_spline(self.xx, yy, k=3, bc_type=('clamped', 'natural'))
    assert_allclose(b1.c, b2.c, atol=1e-15)
    np.random.seed(1234)
    k, n = (3, 22)
    x = np.sort(np.random.random(size=n))
    y = np.random.random(size=(n, 5, 6, 7))
    d_l = [(1, np.zeros((5, 6, 7)))]
    d_r = [(1, np.zeros((5, 6, 7)))]
    b1 = make_interp_spline(x, y, k, bc_type=(d_l, d_r))
    b2 = make_interp_spline(x, y, k, bc_type='clamped')
    assert_allclose(b1.c, b2.c, atol=1e-15)