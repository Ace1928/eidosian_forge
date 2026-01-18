import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
@mpmath_check('0.10')
def test_sos_freqz_against_mp(self):
    from . import mpsig
    N = 500
    order = 25
    Wn = 0.15
    with mpmath.workdps(80):
        z_mp, p_mp, k_mp = mpsig.butter_lp(order, Wn)
        w_mp, h_mp = mpsig.zpkfreqz(z_mp, p_mp, k_mp, N)
    w_mp = np.array([float(x) for x in w_mp])
    h_mp = np.array([complex(x) for x in h_mp])
    sos = butter(order, Wn, output='sos')
    w, h = sosfreqz(sos, worN=N)
    assert_allclose(w, w_mp, rtol=1e-12, atol=1e-14)
    assert_allclose(h, h_mp, rtol=1e-12, atol=1e-14)