import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_nyquist(self):
    w, h = freqz([1.0], worN=8, include_nyquist=True)
    assert_array_almost_equal(w, np.pi * np.arange(8) / 7.0)
    assert_array_almost_equal(h, np.ones(8))
    w, h = freqz([1.0], worN=9, include_nyquist=True)
    assert_array_almost_equal(w, np.pi * np.arange(9) / 8.0)
    assert_array_almost_equal(h, np.ones(9))
    for a in [1, np.ones(2)]:
        w, h = freqz(np.ones(2), a, worN=0, include_nyquist=True)
        assert_equal(w.shape, (0,))
        assert_equal(h.shape, (0,))
        assert_equal(h.dtype, np.dtype('complex128'))
    w1, h1 = freqz([1.0], worN=8, whole=True, include_nyquist=True)
    w2, h2 = freqz([1.0], worN=8, whole=True, include_nyquist=False)
    assert_array_almost_equal(w1, w2)
    assert_array_almost_equal(h1, h2)