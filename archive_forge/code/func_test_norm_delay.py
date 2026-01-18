import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_norm_delay(self):
    for N in (1, 2, 3, 4, 5, 51, 72):
        for w0 in (1, 100):
            b, a = bessel(N, w0, analog=True, norm='delay')
            w = np.linspace(0, 10 * w0, 1000)
            w, h = freqs(b, a, w)
            delay = -np.diff(np.unwrap(np.angle(h))) / np.diff(w)
            assert_allclose(delay[0], 1 / w0, rtol=0.0001)