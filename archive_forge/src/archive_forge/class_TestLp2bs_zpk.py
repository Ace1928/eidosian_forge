import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestLp2bs_zpk:

    def test_basic(self):
        z = [-2j, +2j]
        p = [-0.75, -0.5 - 0.5j, -0.5 + 0.5j]
        k = 3
        z_bs, p_bs, k_bs = lp2bs_zpk(z, p, k, 35, 12)
        assert_allclose(sort(z_bs), sort([+35j, -35j, +3j + sqrt(1234) * 1j, -3j + sqrt(1234) * 1j, +3j - sqrt(1234) * 1j, -3j - sqrt(1234) * 1j]))
        assert_allclose(sort(p_bs), sort([+3j * sqrt(129) - 8, -3j * sqrt(129) - 8, -6 + 6j - sqrt(-1225 - 72j), -6 - 6j - sqrt(-1225 + 72j), -6 + 6j + sqrt(-1225 - 72j), -6 - 6j + sqrt(-1225 + 72j)]))
        assert_allclose(k_bs, 32)