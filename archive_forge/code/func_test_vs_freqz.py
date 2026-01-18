import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_vs_freqz(self):
    b, a = cheby1(4, 5, 0.5, analog=False, output='ba')
    z, p, k = cheby1(4, 5, 0.5, analog=False, output='zpk')
    w1, h1 = freqz(b, a)
    w2, h2 = freqz_zpk(z, p, k)
    assert_allclose(w1, w2)
    assert_allclose(h1, h2, rtol=1e-06)