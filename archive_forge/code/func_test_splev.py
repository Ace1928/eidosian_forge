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
def test_splev(self):
    xnew, b, b2 = (self.xnew, self.b, self.b2)
    assert_allclose(splev(xnew, b), b(xnew), atol=1e-15, rtol=1e-15)
    assert_allclose(splev(xnew, b.tck), b(xnew), atol=1e-15, rtol=1e-15)
    assert_allclose([splev(x, b) for x in xnew], b(xnew), atol=1e-15, rtol=1e-15)
    with assert_raises(ValueError, match='Calling splev.. with BSpline'):
        splev(xnew, b2)
    sh = tuple(range(1, b2.c.ndim)) + (0,)
    cc = b2.c.transpose(sh)
    tck = (b2.t, cc, b2.k)
    assert_allclose(splev(xnew, tck), b2(xnew).transpose(sh), atol=1e-15, rtol=1e-15)