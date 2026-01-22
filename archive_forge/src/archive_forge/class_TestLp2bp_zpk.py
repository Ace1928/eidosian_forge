import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestLp2bp_zpk:

    def test_basic(self):
        z = [-2j, +2j]
        p = [-0.75, -0.5 - 0.5j, -0.5 + 0.5j]
        k = 3
        z_bp, p_bp, k_bp = lp2bp_zpk(z, p, k, 15, 8)
        assert_allclose(sort(z_bp), sort([-25j, -9j, 0, +9j, +25j]))
        assert_allclose(sort(p_bp), sort([-3 + 6j * sqrt(6), -3 - 6j * sqrt(6), +2j + sqrt(-8j - 225) - 2, -2j + sqrt(+8j - 225) - 2, +2j - sqrt(-8j - 225) - 2, -2j - sqrt(+8j - 225) - 2]))
        assert_allclose(k_bp, 24)