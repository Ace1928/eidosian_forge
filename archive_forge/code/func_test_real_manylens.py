import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from itertools import product
from math import gcd
import pytest
from pytest import raises as assert_raises
from numpy.testing import (
from numpy import array, arange
import numpy as np
from scipy.fft import fft
from scipy.ndimage import correlate1d
from scipy.optimize import fmin, linear_sum_assignment
from scipy import signal
from scipy.signal import (
from scipy.signal.windows import hann
from scipy.signal._signaltools import (_filtfilt_gust, _compute_factors,
from scipy.signal._upfirdn import _upfirdn_modes
from scipy._lib import _testutils
from scipy._lib._util import ComplexWarning, np_long, np_ulong
@pytest.mark.slow()
@pytest.mark.parametrize('shape_a_0, shape_b_0', gen_oa_shapes_eq(list(range(100)) + list(range(100, 1000, 23))))
def test_real_manylens(self, shape_a_0, shape_b_0):
    a = np.random.rand(shape_a_0)
    b = np.random.rand(shape_b_0)
    expected = fftconvolve(a, b)
    out = oaconvolve(a, b)
    assert_array_almost_equal(out, expected)