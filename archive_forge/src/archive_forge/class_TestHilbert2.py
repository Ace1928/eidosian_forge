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
class TestHilbert2:

    def test_bad_args(self):
        x = np.array([[1.0 + 0j]])
        assert_raises(ValueError, hilbert2, x)
        x = np.arange(24).reshape(2, 3, 4)
        assert_raises(ValueError, hilbert2, x)
        x = np.arange(16).reshape(4, 4)
        assert_raises(ValueError, hilbert2, x, N=0)
        assert_raises(ValueError, hilbert2, x, N=(2, 0))
        assert_raises(ValueError, hilbert2, x, N=(2,))

    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_hilbert2_types(self, dtype):
        in_typed = np.zeros((2, 32), dtype=dtype)
        assert_equal(np.real(signal.hilbert2(in_typed)).dtype, dtype)