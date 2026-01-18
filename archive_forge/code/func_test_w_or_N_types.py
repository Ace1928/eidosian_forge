import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_w_or_N_types(self):
    for N in (8, np.int8(8), np.int16(8), np.int32(8), np.int64(8), np.array(8)):
        w, gd = group_delay((1, 1), N)
        assert_array_almost_equal(w, pi * np.arange(8) / 8)
        assert_array_almost_equal(gd, np.zeros(8))
    for w in (8.0, 8.0 + 0j):
        w_out, gd = group_delay((1, 1), w)
        assert_array_almost_equal(w_out, [8])
        assert_array_almost_equal(gd, [0])