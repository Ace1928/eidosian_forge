import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestTf2Sos:

    def test_basic(self):
        num = [2, 16, 44, 56, 32]
        den = [3, 3, -15, 18, -12]
        sos = tf2sos(num, den)
        sos2 = [[0.6667, 4.0, 5.3333, 1.0, +2.0, -4.0], [1.0, 2.0, 2.0, 1.0, -1.0, +1.0]]
        assert_array_almost_equal(sos, sos2, decimal=4)
        b = [1, -3, 11, -27, 18]
        a = [16, 12, 2, -4, -1]
        sos = tf2sos(b, a)
        sos2 = [[0.0625, -0.1875, 0.125, 1.0, -0.25, -0.125], [1.0, +0.0, 9.0, 1.0, +1.0, +0.5]]

    @pytest.mark.parametrize('b, a, analog, sos', [([1], [1], False, [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]), ([1], [1], True, [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]]), ([1], [1.0, 0.0, -1.01, 0, 0.01], False, [[1.0, 0.0, 0.0, 1.0, 0.0, -0.01], [1.0, 0.0, 0.0, 1.0, 0.0, -1]]), ([1], [1.0, 0.0, -1.01, 0, 0.01], True, [[0.0, 0.0, 1.0, 1.0, 0.0, -1], [0.0, 0.0, 1.0, 1.0, 0.0, -0.01]])])
    def test_analog(self, b, a, analog, sos):
        sos2 = tf2sos(b, a, analog=analog)
        assert_array_almost_equal(sos, sos2, decimal=4)