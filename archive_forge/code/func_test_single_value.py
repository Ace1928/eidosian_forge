from numpy.testing import (assert_, assert_equal, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
from numpy import mgrid, pi, sin, ogrid, poly1d, linspace
import numpy as np
from scipy.interpolate import (interp1d, interp2d, lagrange, PPoly, BPoly,
from scipy.special import poch, gamma
from scipy.interpolate import _ppoly
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
from scipy.integrate import nquad
from scipy.special import binom
@pytest.mark.parametrize('kind', ('linear', 'nearest', 'nearest-up', 'previous', 'next'))
def test_single_value(self, kind):
    f = interp1d([1.5], [6], kind=kind, bounds_error=False, fill_value=(2, 10))
    assert_array_equal(f([1, 1.5, 2]), [2, 6, 10])
    f = interp1d([1.5], [6], kind=kind, bounds_error=True)
    with assert_raises(ValueError, match='x_new is above'):
        f(2.0)