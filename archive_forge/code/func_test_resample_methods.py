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
@pytest.mark.parametrize('method, ext, padtype', [('fft', False, None)] + list(product(['polyphase'], [False, True], padtype_options)))
def test_resample_methods(self, method, ext, padtype):
    rate = 100
    rates_to = [49, 50, 51, 99, 100, 101, 199, 200, 201]
    t = np.arange(rate) / float(rate)
    freqs = np.array((1.0, 10.0, 40.0))[:, np.newaxis]
    x = np.sin(2 * np.pi * freqs * t) * hann(rate)
    for rate_to in rates_to:
        t_to = np.arange(rate_to) / float(rate_to)
        y_tos = np.sin(2 * np.pi * freqs * t_to) * hann(rate_to)
        if method == 'fft':
            y_resamps = signal.resample(x, rate_to, axis=-1)
        else:
            if ext and rate_to != rate:
                g = gcd(rate_to, rate)
                up = rate_to // g
                down = rate // g
                max_rate = max(up, down)
                f_c = 1.0 / max_rate
                half_len = 10 * max_rate
                window = signal.firwin(2 * half_len + 1, f_c, window=('kaiser', 5.0))
                polyargs = {'window': window, 'padtype': padtype}
            else:
                polyargs = {'padtype': padtype}
            y_resamps = signal.resample_poly(x, rate_to, rate, axis=-1, **polyargs)
        for y_to, y_resamp, freq in zip(y_tos, y_resamps, freqs):
            if freq >= 0.5 * rate_to:
                y_to.fill(0.0)
                if padtype in ['minimum', 'maximum']:
                    assert_allclose(y_resamp, y_to, atol=0.3)
                else:
                    assert_allclose(y_resamp, y_to, atol=0.001)
            else:
                assert_array_equal(y_to.shape, y_resamp.shape)
                corr = np.corrcoef(y_to, y_resamp)[0, 1]
                assert_(corr > 0.99, msg=(corr, rate, rate_to))
    rng = np.random.RandomState(0)
    x = hann(rate) * np.cumsum(rng.randn(rate))
    for rate_to in rates_to:
        t_to = np.arange(rate_to) / float(rate_to)
        y_to = np.interp(t_to, t, x)
        if method == 'fft':
            y_resamp = signal.resample(x, rate_to)
        else:
            y_resamp = signal.resample_poly(x, rate_to, rate, padtype=padtype)
        assert_array_equal(y_to.shape, y_resamp.shape)
        corr = np.corrcoef(y_to, y_resamp)[0, 1]
        assert_(corr > 0.99, msg=corr)
    if method == 'fft':
        x1 = np.array([1.0 + 0j, 0.0 + 0j])
        y1_test = signal.resample(x1, 4)
        y1_true = np.array([1.0 + 0j, 0.5 + 0j, 0.0 + 0j, 0.5 + 0j])
        assert_allclose(y1_test, y1_true, atol=1e-12)
        x2 = np.array([1.0, 0.5, 0.0, 0.5])
        y2_test = signal.resample(x2, 2)
        y2_true = np.array([1.0, 0.0])
        assert_allclose(y2_test, y2_true, atol=1e-12)