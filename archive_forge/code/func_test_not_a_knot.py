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
def test_not_a_knot(self):
    for k in [3, 5]:
        b = make_interp_spline(self.xx, self.yy, k)
        assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)