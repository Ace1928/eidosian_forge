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
def test_tck(self):
    b = _make_random_spline()
    tck = b.tck
    assert_allclose(b.t, tck[0], atol=1e-15, rtol=1e-15)
    assert_allclose(b.c, tck[1], atol=1e-15, rtol=1e-15)
    assert_equal(b.k, tck[2])
    with pytest.raises(AttributeError):
        b.tck = 'foo'