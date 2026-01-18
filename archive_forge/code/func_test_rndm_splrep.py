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
def test_rndm_splrep(self):
    np.random.seed(1234)
    x = np.sort(np.random.random(20))
    y = np.random.random(20)
    tck = splrep(x, y)
    b = BSpline(*tck)
    t, k = (b.t, b.k)
    xx = np.linspace(t[k], t[-k - 1], 80)
    assert_allclose(b(xx), splev(xx, tck), atol=1e-14)