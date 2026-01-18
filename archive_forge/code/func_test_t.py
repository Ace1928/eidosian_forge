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
def test_t():
    """Verify that the times of the slices are correct. """
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=2)
    assert SFT.T == 1 / 2
    assert SFT.fs == 2.0
    assert SFT.delta_t == 4 * 1 / 2
    t_stft = np.arange(0, SFT.p_max(10)) * SFT.delta_t
    assert_equal(SFT.t(10), t_stft)
    assert_equal(SFT.t(10, 1, 3), t_stft[1:3])
    SFT.T = 1 / 4
    assert SFT.T == 1 / 4
    assert SFT.fs == 4
    SFT.fs = 1 / 8
    assert SFT.fs == 1 / 8
    assert SFT.T == 8