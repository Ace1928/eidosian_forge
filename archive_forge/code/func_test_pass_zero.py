import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
@pytest.mark.parametrize('ftype,pass_zero,peak,notch', [('peak', True, 123.45, 61.725), ('peak', False, 61.725, 123.45), ('peak', None, 61.725, 123.45), ('notch', None, 61.725, 123.45), ('notch', True, 123.45, 61.725), ('notch', False, 61.725, 123.45)])
def test_pass_zero(self, ftype, pass_zero, peak, notch):
    b, a = iircomb(123.45, 30, ftype=ftype, fs=1234.5, pass_zero=pass_zero)
    freqs, response = freqz(b, a, [peak, notch], fs=1234.5)
    assert abs(response[0]) > 0.99
    assert abs(response[1]) < 1e-10