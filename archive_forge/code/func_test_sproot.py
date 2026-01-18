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
def test_sproot(self):
    b, b2 = (self.b, self.b2)
    roots = np.array([0.5, 1.5, 2.5, 3.5]) * np.pi
    assert_allclose(sproot(b), roots, atol=1e-07, rtol=1e-07)
    assert_allclose(sproot((b.t, b.c, b.k)), roots, atol=1e-07, rtol=1e-07)
    with assert_raises(ValueError, match='Calling sproot.. with BSpline'):
        sproot(b2, mest=50)
    c2r = b2.c.transpose(1, 2, 0)
    rr = np.asarray(sproot((b2.t, c2r, b2.k), mest=50))
    assert_equal(rr.shape, (3, 2, 4))
    assert_allclose(rr - roots, 0, atol=1e-12)