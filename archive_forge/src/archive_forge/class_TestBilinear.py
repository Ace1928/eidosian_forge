import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestBilinear:

    def test_basic(self):
        b = [0.14879732743343033]
        a = [1, 0.5455223688052221, 0.14879732743343033]
        b_z, a_z = bilinear(b, a, 0.5)
        assert_array_almost_equal(b_z, [0.087821, 0.17564, 0.087821], decimal=5)
        assert_array_almost_equal(a_z, [1, -1.0048, 0.35606], decimal=4)
        b = [1, 0, 0.17407467530697837]
        a = [1, 0.1846057532615225, 0.17407467530697837]
        b_z, a_z = bilinear(b, a, 0.5)
        assert_array_almost_equal(b_z, [0.86413, -1.2158, 0.86413], decimal=4)
        assert_array_almost_equal(a_z, [1, -1.2158, 0.72826], decimal=4)