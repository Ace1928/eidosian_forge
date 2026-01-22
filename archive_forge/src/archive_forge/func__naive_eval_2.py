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
def _naive_eval_2(x, t, c, k):
    """Naive B-spline evaluation, another way."""
    n = len(t) - (k + 1)
    assert n >= k + 1
    assert len(c) >= n
    assert t[k] <= x <= t[n]
    return sum((c[i] * _naive_B(x, k, i, t) for i in range(n)))