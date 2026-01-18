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
def test_quintic_derivs(self):
    k, n = (5, 7)
    x = np.arange(n).astype(np.float64)
    y = np.sin(x)
    der_l = [(1, -12.0), (2, 1)]
    der_r = [(1, 8.0), (2, 3.0)]
    b = make_interp_spline(x, y, k=k, bc_type=(der_l, der_r))
    assert_allclose(b(x), y, atol=1e-14, rtol=1e-14)
    assert_allclose([b(x[0], 1), b(x[0], 2)], [val for nu, val in der_l])
    assert_allclose([b(x[-1], 1), b(x[-1], 2)], [val for nu, val in der_r])