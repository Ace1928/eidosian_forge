import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
class TestFirwin2:

    def test_invalid_args(self):
        with assert_raises(ValueError, match='must be of same length'):
            firwin2(50, [0, 0.5, 1], [0.0, 1.0])
        with assert_raises(ValueError, match='ntaps must be less than nfreqs'):
            firwin2(50, [0, 0.5, 1], [0.0, 1.0, 1.0], nfreqs=33)
        with assert_raises(ValueError, match='must be nondecreasing'):
            firwin2(50, [0, 0.5, 0.4, 1.0], [0, 0.25, 0.5, 1.0])
        with assert_raises(ValueError, match='must not occur more than twice'):
            firwin2(50, [0, 0.1, 0.1, 0.1, 1.0], [0.0, 0.5, 0.75, 1.0, 1.0])
        with assert_raises(ValueError, match='start with 0'):
            firwin2(50, [0.5, 1.0], [0.0, 1.0])
        with assert_raises(ValueError, match='end with fs/2'):
            firwin2(50, [0.0, 0.5], [0.0, 1.0])
        with assert_raises(ValueError, match='0 must not be repeated'):
            firwin2(50, [0.0, 0.0, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0])
        with assert_raises(ValueError, match='fs/2 must not be repeated'):
            firwin2(50, [0.0, 0.5, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0])
        with assert_raises(ValueError, match='cannot contain numbers that are too close'):
            firwin2(50, [0.0, 0.5 - np.finfo(float).eps * 0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0])
        with assert_raises(ValueError, match='Type II filter'):
            firwin2(16, [0.0, 0.5, 1.0], [0.0, 1.0, 1.0])
        with assert_raises(ValueError, match='Type III filter'):
            firwin2(17, [0.0, 0.5, 1.0], [0.0, 1.0, 1.0], antisymmetric=True)
        with assert_raises(ValueError, match='Type III filter'):
            firwin2(17, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0], antisymmetric=True)
        with assert_raises(ValueError, match='Type III filter'):
            firwin2(17, [0.0, 0.5, 1.0], [1.0, 1.0, 1.0], antisymmetric=True)
        with assert_raises(ValueError, match='Type IV filter'):
            firwin2(16, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0], antisymmetric=True)

    def test01(self):
        width = 0.04
        beta = 12.0
        ntaps = 400
        freq = [0.0, 0.5, 1.0]
        gain = [1.0, 1.0, 0.0]
        taps = firwin2(ntaps, freq, gain, window=('kaiser', beta))
        freq_samples = np.array([0.0, 0.25, 0.5 - width / 2, 0.5 + width / 2, 0.75, 1.0 - width / 2])
        freqs, response = freqz(taps, worN=np.pi * freq_samples)
        assert_array_almost_equal(np.abs(response), [1.0, 1.0, 1.0, 1.0 - width, 0.5, width], decimal=5)

    def test02(self):
        width = 0.04
        beta = 12.0
        ntaps = 401
        freq = [0.0, 0.5, 0.5, 1.0]
        gain = [0.0, 0.0, 1.0, 1.0]
        taps = firwin2(ntaps, freq, gain, window=('kaiser', beta))
        freq_samples = np.array([0.0, 0.25, 0.5 - width, 0.5 + width, 0.75, 1.0])
        freqs, response = freqz(taps, worN=np.pi * freq_samples)
        assert_array_almost_equal(np.abs(response), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)

    def test03(self):
        width = 0.02
        ntaps, beta = kaiserord(120, width)
        ntaps = int(ntaps) | 1
        freq = [0.0, 0.4, 0.4, 0.5, 0.5, 1.0]
        gain = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        taps = firwin2(ntaps, freq, gain, window=('kaiser', beta))
        freq_samples = np.array([0.0, 0.4 - width, 0.4 + width, 0.45, 0.5 - width, 0.5 + width, 0.75, 1.0])
        freqs, response = freqz(taps, worN=np.pi * freq_samples)
        assert_array_almost_equal(np.abs(response), [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)

    def test04(self):
        """Test firwin2 when window=None."""
        ntaps = 5
        freq = [0.0, 0.5, 0.5, 1.0]
        gain = [1.0, 1.0, 0.0, 0.0]
        taps = firwin2(ntaps, freq, gain, window=None, nfreqs=8193)
        alpha = 0.5 * (ntaps - 1)
        m = np.arange(0, ntaps) - alpha
        h = 0.5 * sinc(0.5 * m)
        assert_array_almost_equal(h, taps)

    def test05(self):
        """Test firwin2 for calculating Type IV filters"""
        ntaps = 1500
        freq = [0.0, 1.0]
        gain = [0.0, 1.0]
        taps = firwin2(ntaps, freq, gain, window=None, antisymmetric=True)
        assert_array_almost_equal(taps[:ntaps // 2], -taps[ntaps // 2:][::-1])
        freqs, response = freqz(taps, worN=2048)
        assert_array_almost_equal(abs(response), freqs / np.pi, decimal=4)

    def test06(self):
        """Test firwin2 for calculating Type III filters"""
        ntaps = 1501
        freq = [0.0, 0.5, 0.55, 1.0]
        gain = [0.0, 0.5, 0.0, 0.0]
        taps = firwin2(ntaps, freq, gain, window=None, antisymmetric=True)
        assert_equal(taps[ntaps // 2], 0.0)
        assert_array_almost_equal(taps[:ntaps // 2], -taps[ntaps // 2 + 1:][::-1])
        freqs, response1 = freqz(taps, worN=2048)
        response2 = np.interp(freqs / np.pi, freq, gain)
        assert_array_almost_equal(abs(response1), response2, decimal=3)

    def test_fs_nyq(self):
        taps1 = firwin2(80, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
        taps2 = firwin2(80, [0.0, 30.0, 60.0], [1.0, 1.0, 0.0], fs=120.0)
        assert_array_almost_equal(taps1, taps2)
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Keyword argument 'nyq'")
            taps2 = firwin2(80, [0.0, 30.0, 60.0], [1.0, 1.0, 0.0], nyq=60.0)
        assert_array_almost_equal(taps1, taps2)

    def test_tuple(self):
        taps1 = firwin2(150, (0.0, 0.5, 0.5, 1.0), (1.0, 1.0, 0.0, 0.0))
        taps2 = firwin2(150, [0.0, 0.5, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0])
        assert_array_almost_equal(taps1, taps2)

    def test_input_modyfication(self):
        freq1 = np.array([0.0, 0.5, 0.5, 1.0])
        freq2 = np.array(freq1)
        firwin2(80, freq1, [1.0, 1.0, 0.0, 0.0])
        assert_equal(freq1, freq2)

    def test_firwin2_deprecations(self):
        with pytest.deprecated_call(match="argument 'nyq' is deprecated"):
            firwin2(1, [0, 10], [1, 1], nyq=10)
        with pytest.deprecated_call(match='use keyword arguments'):
            firwin2(5, [0.0, 0.5, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0], 8193, None)