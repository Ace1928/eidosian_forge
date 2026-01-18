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
def test_splantider(self):
    for b in [self.b, self.b2]:
        ct = len(b.t) - len(b.c)
        if ct > 0:
            b.c = np.r_[b.c, np.zeros((ct,) + b.c.shape[1:])]
        for n in [1, 2, 3]:
            bd = splantider(b)
            tck_d = _impl.splantider((b.t, b.c, b.k))
            assert_allclose(bd.t, tck_d[0], atol=1e-15)
            assert_allclose(bd.c, tck_d[1], atol=1e-15)
            assert_equal(bd.k, tck_d[2])
            assert_(isinstance(bd, BSpline))
            assert_(isinstance(tck_d, tuple))