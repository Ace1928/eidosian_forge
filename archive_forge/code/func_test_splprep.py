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
def test_splprep(self):
    x = np.arange(15).reshape((3, 5))
    b, u = splprep(x)
    tck, u1 = _impl.splprep(x)
    assert_allclose(u, u1, atol=1e-15)
    assert_allclose(splev(u, b), x, atol=1e-15)
    assert_allclose(splev(u, tck), x, atol=1e-15)
    (b_f, u_f), _, _, _ = splprep(x, s=0, full_output=True)
    assert_allclose(u, u_f, atol=1e-15)
    assert_allclose(splev(u_f, b_f), x, atol=1e-15)