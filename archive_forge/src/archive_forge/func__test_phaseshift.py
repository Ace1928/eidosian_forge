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
def _test_phaseshift(self, method, zero_phase):
    rate = 120
    rates_to = [15, 20, 30, 40]
    t_tot = 100
    t = np.arange(rate * t_tot + 1) / float(rate)
    freqs = np.array(rates_to) * 0.8 / 2
    d = np.exp(1j * 2 * np.pi * freqs[:, np.newaxis] * t) * signal.windows.tukey(t.size, 0.1)
    for rate_to in rates_to:
        q = rate // rate_to
        t_to = np.arange(rate_to * t_tot + 1) / float(rate_to)
        d_tos = np.exp(1j * 2 * np.pi * freqs[:, np.newaxis] * t_to) * signal.windows.tukey(t_to.size, 0.1)
        if method == 'fir':
            n = 30
            system = signal.dlti(signal.firwin(n + 1, 1.0 / q, window='hamming'), 1.0)
        elif method == 'iir':
            n = 8
            wc = 0.8 * np.pi / q
            system = signal.dlti(*signal.cheby1(n, 0.05, wc / np.pi))
        if zero_phase is False:
            _, h_resps = signal.freqz(system.num, system.den, freqs / rate * 2 * np.pi)
            h_resps /= np.abs(h_resps)
        else:
            h_resps = np.ones_like(freqs)
        y_resamps = signal.decimate(d.real, q, n, ftype=system, zero_phase=zero_phase)
        h_resamps = np.sum(d_tos.conj() * y_resamps, axis=-1)
        h_resamps /= np.abs(h_resamps)
        subnyq = freqs < 0.5 * rate_to
        assert_allclose(np.angle(h_resps.conj() * h_resamps)[subnyq], 0, atol=0.001, rtol=0.001)