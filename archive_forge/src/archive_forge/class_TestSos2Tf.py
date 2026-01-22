import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestSos2Tf:

    def test_basic(self):
        sos = [[1, 1, 1, 1, 0, -1], [-2, 3, 1, 1, 10, 1]]
        b, a = sos2tf(sos)
        assert_array_almost_equal(b, [-2, 1, 2, 4, 1])
        assert_array_almost_equal(a, [1, 10, 0, -10, -1])