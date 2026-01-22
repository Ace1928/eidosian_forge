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
def B_0123(x, der=0):
    """A quadratic B-spline function B(x | 0, 1, 2, 3)."""
    x = np.atleast_1d(x)
    conds = [x < 1, (x > 1) & (x < 2), x > 2]
    if der == 0:
        funcs = [lambda x: x * x / 2.0, lambda x: 3.0 / 4 - (x - 3.0 / 2) ** 2, lambda x: (3.0 - x) ** 2 / 2]
    elif der == 2:
        funcs = [lambda x: 1.0, lambda x: -2.0, lambda x: 1.0]
    else:
        raise ValueError('never be here: der=%s' % der)
    pieces = np.piecewise(x, conds, funcs)
    return pieces