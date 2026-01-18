import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_tolerances(self):
    eps = spacing(1)
    assert_allclose(_cplxpair([1j, -1j, 1 + 1j * eps], tol=2 * eps), [-1j, 1j, 1 + 1j * eps])
    assert_allclose(_cplxpair([-eps + 1j, +eps - 1j]), [-1j, +1j])
    assert_allclose(_cplxpair([+eps + 1j, -eps - 1j]), [-1j, +1j])
    assert_allclose(_cplxpair([+1j, -1j]), [-1j, +1j])