import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestBilinear_zpk:

    def test_basic(self):
        z = [-2j, +2j]
        p = [-0.75, -0.5 - 0.5j, -0.5 + 0.5j]
        k = 3
        z_d, p_d, k_d = bilinear_zpk(z, p, k, 10)
        assert_allclose(sort(z_d), sort([(20 - 2j) / (20 + 2j), (20 + 2j) / (20 - 2j), -1]))
        assert_allclose(sort(p_d), sort([77 / 83, (1j / 2 + 39 / 2) / (41 / 2 - 1j / 2), (39 / 2 - 1j / 2) / (1j / 2 + 41 / 2)]))
        assert_allclose(k_d, 9696 / 69803)