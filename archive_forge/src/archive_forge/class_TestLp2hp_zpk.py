import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestLp2hp_zpk:

    def test_basic(self):
        z = []
        p = [(-1 + 1j) / np.sqrt(2), (-1 - 1j) / np.sqrt(2)]
        k = 1
        z_hp, p_hp, k_hp = lp2hp_zpk(z, p, k, 5)
        assert_array_equal(z_hp, [0, 0])
        assert_allclose(sort(p_hp), sort(p) * 5)
        assert_allclose(k_hp, 1)
        z = [-2j, +2j]
        p = [-0.75, -0.5 - 0.5j, -0.5 + 0.5j]
        k = 3
        z_hp, p_hp, k_hp = lp2hp_zpk(z, p, k, 6)
        assert_allclose(sort(z_hp), sort([-3j, 0, +3j]))
        assert_allclose(sort(p_hp), sort([-8, -6 - 6j, -6 + 6j]))
        assert_allclose(k_hp, 32)