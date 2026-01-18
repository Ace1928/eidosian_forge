import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_norm_phase(self):
    for N in (1, 2, 3, 4, 5, 51, 72):
        for w0 in (1, 100):
            b, a = bessel(N, w0, analog=True, norm='phase')
            w = np.linspace(0, w0, 100)
            w, h = freqs(b, a, w)
            phase = np.unwrap(np.angle(h))
            assert_allclose(phase[[0, -1]], (0, -N * pi / 4), rtol=0.1)