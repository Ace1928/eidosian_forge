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
def test_splrep_errors(self):
    x, y = (self.xx, self.yy)
    y2 = np.c_[y, y]
    with assert_raises(ValueError):
        splrep(x, y2)
    with assert_raises(ValueError):
        _impl.splrep(x, y2)
    with assert_raises(TypeError, match='m > k must hold'):
        splrep(x[:3], y[:3])
    with assert_raises(TypeError, match='m > k must hold'):
        _impl.splrep(x[:3], y[:3])