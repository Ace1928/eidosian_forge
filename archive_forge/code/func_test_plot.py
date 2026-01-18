import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_plot(self):

    def plot(w, h):
        assert_array_almost_equal(w, np.pi * np.arange(8.0) / 8)
        assert_array_almost_equal(h, np.ones(8))
    assert_raises(ZeroDivisionError, freqz, [1.0], worN=8, plot=lambda w, h: 1 / 0)
    freqz([1.0], worN=8, plot=plot)