import math
from itertools import product
from typing import cast, get_args, Literal
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from scipy.fft import fftshift
from scipy.stats import norm as normal_distribution  # type: ignore
from scipy.signal import get_window, welch, stft, istft, spectrogram
from scipy.signal._short_time_fft import FFT_MODE_TYPE, \
from scipy.signal.windows import gaussian
def test_spectrogram():
    """Verify spectrogram and cross-spectrogram methods. """
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=1)
    x, y = (np.ones(10), np.arange(10))
    X, Y = (SFT.stft(x), SFT.stft(y))
    assert_allclose(SFT.spectrogram(x), X.real ** 2 + X.imag ** 2)
    assert_allclose(SFT.spectrogram(x, y), X * Y.conj())