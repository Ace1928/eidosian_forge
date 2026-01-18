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
def test_rank_3_FIR_init_cond(self):
    x = self.generate((4, 3, 2))
    b = self.convert_dtype([1, 0, -1])
    a = self.convert_dtype([1])
    for axis in range(x.ndim):
        zi_shape = list(x.shape)
        zi_shape[axis] = 2
        zi = self.convert_dtype(np.ones(zi_shape))
        zi1 = self.convert_dtype([1, 1])
        y, zf = lfilter(b, a, x, axis, zi)

        def lf0(w):
            return lfilter(b, a, w, zi=zi1)[0]

        def lf1(w):
            return lfilter(b, a, w, zi=zi1)[1]
        y_r = np.apply_along_axis(lf0, axis, x)
        zf_r = np.apply_along_axis(lf1, axis, x)
        assert_array_almost_equal(y, y_r)
        assert_array_almost_equal(zf, zf_r)