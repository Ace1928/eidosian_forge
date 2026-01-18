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
def test_tx_neq_ty(self):
    x = np.arange(6)
    y = np.arange(7) + 1.5
    spl_x = make_interp_spline(x, x ** 3, k=3)
    spl_y = make_interp_spline(y, y ** 2 + 2 * y, k=3)
    cc = spl_x.c[:, None] * spl_y.c[None, :]
    bspl = NdBSpline((spl_x.t, spl_y.t), cc, (spl_x.k, spl_y.k))
    values = (x ** 3)[:, None] * (y ** 2 + 2 * y)[None, :]
    rgi = RegularGridInterpolator((x, y), values)
    xi = [(a, b) for a, b in itertools.product(x, y)]
    bxi = bspl(xi)
    assert not np.isnan(bxi).any()
    assert_allclose(bxi, rgi(xi), atol=1e-14)
    assert_allclose(bxi.reshape(values.shape), values, atol=1e-14)