import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestLp2lp_zpk:

    def test_basic(self):
        z = []
        p = [(-1 + 1j) / np.sqrt(2), (-1 - 1j) / np.sqrt(2)]
        k = 1
        z_lp, p_lp, k_lp = lp2lp_zpk(z, p, k, 5)
        assert_array_equal(z_lp, [])
        assert_allclose(sort(p_lp), sort(p) * 5)
        assert_allclose(k_lp, 25)
        z = [-2j, +2j]
        p = [-0.75, -0.5 - 0.5j, -0.5 + 0.5j]
        k = 3
        z_lp, p_lp, k_lp = lp2lp_zpk(z, p, k, 20)
        assert_allclose(sort(z_lp), sort([-40j, +40j]))
        assert_allclose(sort(p_lp), sort([-15, -10 - 10j, -10 + 10j]))
        assert_allclose(k_lp, 60)