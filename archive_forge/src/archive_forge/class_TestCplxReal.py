import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestCplxReal:

    def test_trivial_input(self):
        assert_equal(_cplxreal([]), ([], []))
        assert_equal(_cplxreal(1), ([], [1]))

    def test_output_order(self):
        zc, zr = _cplxreal(np.roots(array([1, 0, 0, 1])))
        assert_allclose(np.append(zc, zr), [1 / 2 + 1j * sin(pi / 3), -1])
        eps = spacing(1)
        a = [0 + 1j, 0 - 1j, eps + 1j, eps - 1j, -eps + 1j, -eps - 1j, 1, 4, 2, 3, 0, 0, 2 + 3j, 2 - 3j, 1 - eps + 1j, 1 + 2j, 1 - 2j, 1 + eps - 1j, 3 + 1j, 3 + 1j, 3 + 1j, 3 - 1j, 3 - 1j, 3 - 1j, 2 - 3j, 2 + 3j]
        zc, zr = _cplxreal(a)
        assert_allclose(zc, [1j, 1j, 1j, 1 + 1j, 1 + 2j, 2 + 3j, 2 + 3j, 3 + 1j, 3 + 1j, 3 + 1j])
        assert_allclose(zr, [0, 0, 1, 2, 3, 4])
        z = array([1 - eps + 1j, 1 + 2j, 1 - 2j, 1 + eps - 1j, 1 + eps + 3j, 1 - 2 * eps - 3j, 0 + 1j, 0 - 1j, 2 + 4j, 2 - 4j, 2 + 3j, 2 - 3j, 3 + 7j, 3 - 7j, 4 - eps + 1j, 4 + eps - 2j, 4 - 1j, 4 - eps + 2j])
        zc, zr = _cplxreal(z)
        assert_allclose(zc, [1j, 1 + 1j, 1 + 2j, 1 + 3j, 2 + 3j, 2 + 4j, 3 + 7j, 4 + 1j, 4 + 2j])
        assert_equal(zr, [])

    def test_unmatched_conjugates(self):
        assert_raises(ValueError, _cplxreal, [1 + 3j, 1 - 3j, 1 + 2j])
        assert_raises(ValueError, _cplxreal, [1 + 3j, 1 - 3j, 1 + 2j, 1 - 3j])
        assert_raises(ValueError, _cplxreal, [1 + 3j, 1 - 3j, 1 + 3j])
        assert_raises(ValueError, _cplxreal, [1 + 3j])
        assert_raises(ValueError, _cplxreal, [1 - 3j])

    def test_real_integer_input(self):
        zc, zr = _cplxreal([2, 0, 1, 4])
        assert_array_equal(zc, [])
        assert_array_equal(zr, [0, 1, 2, 4])