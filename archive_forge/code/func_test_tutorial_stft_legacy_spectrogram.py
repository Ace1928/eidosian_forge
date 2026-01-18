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
def test_tutorial_stft_legacy_spectrogram():
    """Verify spectrogram example in "Comparison with Legacy Implementation"
    from the "User Guide".

    In :ref:`tutorial_stft_legacy_stft` (file ``signal.rst``) of the
    :ref:`user_guide` the legacy and the new implementation are compared.
    """
    fs, N = (200, 1001)
    t_z = np.arange(N) / fs
    z = np.exp(2j * np.pi * 70 * (t_z - 0.2 * t_z ** 2))
    nperseg, noverlap = (50, 40)
    win = ('gaussian', 0.01 * fs)
    f2_u, t2, Sz2_u = spectrogram(z, fs, win, nperseg, noverlap, detrend=None, return_onesided=False, scaling='spectrum', mode='complex')
    f2, Sz2 = (fftshift(f2_u), fftshift(Sz2_u, axes=0))
    SFT = ShortTimeFFT.from_window(win, fs, nperseg, noverlap, fft_mode='centered', scale_to='magnitude', phase_shift=None)
    Sz3 = SFT.stft(z, p0=0, p1=(N - noverlap) // SFT.hop, k_offset=nperseg // 2)
    t3 = SFT.t(N, p0=0, p1=(N - noverlap) // SFT.hop, k_offset=nperseg // 2)
    assert_allclose(t2, t3)
    assert_allclose(f2, SFT.f)
    assert_allclose(Sz2, Sz3)