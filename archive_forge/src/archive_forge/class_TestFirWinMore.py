import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
class TestFirWinMore:
    """Different author, different style, different tests..."""

    def test_lowpass(self):
        width = 0.04
        ntaps, beta = kaiserord(120, width)
        kwargs = dict(cutoff=0.5, window=('kaiser', beta), scale=False)
        taps = firwin(ntaps, **kwargs)
        assert_array_almost_equal(taps[:ntaps // 2], taps[ntaps:ntaps - ntaps // 2 - 1:-1])
        freq_samples = np.array([0.0, 0.25, 0.5 - width / 2, 0.5 + width / 2, 0.75, 1.0])
        freqs, response = freqz(taps, worN=np.pi * freq_samples)
        assert_array_almost_equal(np.abs(response), [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)
        taps_str = firwin(ntaps, pass_zero='lowpass', **kwargs)
        assert_allclose(taps, taps_str)

    def test_highpass(self):
        width = 0.04
        ntaps, beta = kaiserord(120, width)
        ntaps |= 1
        kwargs = dict(cutoff=0.5, window=('kaiser', beta), scale=False)
        taps = firwin(ntaps, pass_zero=False, **kwargs)
        assert_array_almost_equal(taps[:ntaps // 2], taps[ntaps:ntaps - ntaps // 2 - 1:-1])
        freq_samples = np.array([0.0, 0.25, 0.5 - width / 2, 0.5 + width / 2, 0.75, 1.0])
        freqs, response = freqz(taps, worN=np.pi * freq_samples)
        assert_array_almost_equal(np.abs(response), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)
        taps_str = firwin(ntaps, pass_zero='highpass', **kwargs)
        assert_allclose(taps, taps_str)

    def test_bandpass(self):
        width = 0.04
        ntaps, beta = kaiserord(120, width)
        kwargs = dict(cutoff=[0.3, 0.7], window=('kaiser', beta), scale=False)
        taps = firwin(ntaps, pass_zero=False, **kwargs)
        assert_array_almost_equal(taps[:ntaps // 2], taps[ntaps:ntaps - ntaps // 2 - 1:-1])
        freq_samples = np.array([0.0, 0.2, 0.3 - width / 2, 0.3 + width / 2, 0.5, 0.7 - width / 2, 0.7 + width / 2, 0.8, 1.0])
        freqs, response = freqz(taps, worN=np.pi * freq_samples)
        assert_array_almost_equal(np.abs(response), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)
        taps_str = firwin(ntaps, pass_zero='bandpass', **kwargs)
        assert_allclose(taps, taps_str)

    def test_bandstop_multi(self):
        width = 0.04
        ntaps, beta = kaiserord(120, width)
        kwargs = dict(cutoff=[0.2, 0.5, 0.8], window=('kaiser', beta), scale=False)
        taps = firwin(ntaps, **kwargs)
        assert_array_almost_equal(taps[:ntaps // 2], taps[ntaps:ntaps - ntaps // 2 - 1:-1])
        freq_samples = np.array([0.0, 0.1, 0.2 - width / 2, 0.2 + width / 2, 0.35, 0.5 - width / 2, 0.5 + width / 2, 0.65, 0.8 - width / 2, 0.8 + width / 2, 0.9, 1.0])
        freqs, response = freqz(taps, worN=np.pi * freq_samples)
        assert_array_almost_equal(np.abs(response), [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)
        taps_str = firwin(ntaps, pass_zero='bandstop', **kwargs)
        assert_allclose(taps, taps_str)

    def test_fs_nyq(self):
        """Test the fs and nyq keywords."""
        nyquist = 1000
        width = 40.0
        relative_width = width / nyquist
        ntaps, beta = kaiserord(120, relative_width)
        taps = firwin(ntaps, cutoff=[300, 700], window=('kaiser', beta), pass_zero=False, scale=False, fs=2 * nyquist)
        assert_array_almost_equal(taps[:ntaps // 2], taps[ntaps:ntaps - ntaps // 2 - 1:-1])
        freq_samples = np.array([0.0, 200, 300 - width / 2, 300 + width / 2, 500, 700 - width / 2, 700 + width / 2, 800, 1000])
        freqs, response = freqz(taps, worN=np.pi * freq_samples / nyquist)
        assert_array_almost_equal(np.abs(response), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Keyword argument 'nyq'")
            taps2 = firwin(ntaps, cutoff=[300, 700], window=('kaiser', beta), pass_zero=False, scale=False, nyq=nyquist)
        assert_allclose(taps2, taps)

    def test_bad_cutoff(self):
        """Test that invalid cutoff argument raises ValueError."""
        assert_raises(ValueError, firwin, 99, -0.5)
        assert_raises(ValueError, firwin, 99, 1.5)
        assert_raises(ValueError, firwin, 99, [0, 0.5])
        assert_raises(ValueError, firwin, 99, [0.5, 1])
        assert_raises(ValueError, firwin, 99, [0.1, 0.5, 0.2])
        assert_raises(ValueError, firwin, 99, [0.1, 0.5, 0.5])
        assert_raises(ValueError, firwin, 99, [])
        assert_raises(ValueError, firwin, 99, [[0.1, 0.2], [0.3, 0.4]])
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Keyword argument 'nyq'")
            assert_raises(ValueError, firwin, 99, 50.0, nyq=40)
            assert_raises(ValueError, firwin, 99, [10, 20, 30], nyq=25)
        assert_raises(ValueError, firwin, 99, 50.0, fs=80)
        assert_raises(ValueError, firwin, 99, [10, 20, 30], fs=50)

    def test_even_highpass_raises_value_error(self):
        """Test that attempt to create a highpass filter with an even number
        of taps raises a ValueError exception."""
        assert_raises(ValueError, firwin, 40, 0.5, pass_zero=False)
        assert_raises(ValueError, firwin, 40, [0.25, 0.5])

    def test_bad_pass_zero(self):
        """Test degenerate pass_zero cases."""
        with assert_raises(ValueError, match='pass_zero must be'):
            firwin(41, 0.5, pass_zero='foo')
        with assert_raises(TypeError, match='cannot be interpreted'):
            firwin(41, 0.5, pass_zero=1.0)
        for pass_zero in ('lowpass', 'highpass'):
            with assert_raises(ValueError, match='cutoff must have one'):
                firwin(41, [0.5, 0.6], pass_zero=pass_zero)
        for pass_zero in ('bandpass', 'bandstop'):
            with assert_raises(ValueError, match='must have at least two'):
                firwin(41, [0.5], pass_zero=pass_zero)

    def test_firwin_deprecations(self):
        with pytest.deprecated_call(match="argument 'nyq' is deprecated"):
            firwin(1, 1, nyq=10)
        with pytest.deprecated_call(match='use keyword arguments'):
            firwin(58, 0.1, 0.03)