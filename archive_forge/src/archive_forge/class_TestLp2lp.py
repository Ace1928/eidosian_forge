import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestLp2lp:

    def test_basic(self):
        b = [1]
        a = [1, np.sqrt(2), 1]
        b_lp, a_lp = lp2lp(b, a, 0.3857425662711212)
        assert_array_almost_equal(b_lp, [0.1488], decimal=4)
        assert_array_almost_equal(a_lp, [1, 0.5455, 0.1488], decimal=4)