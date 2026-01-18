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
def test_tutorial_stft_legacy_stft():
    """Verify STFT example in "Comparison with Legacy Implementation" from the
    "User Guide".

    In :ref:`tutorial_stft_legacy_stft` (file ``signal.rst``) of the
    :ref:`user_guide` the legacy and the new implementation are compared.
    """
    fs, N = (200, 1001)
    t_z = np.arange(N) / fs
    z = np.exp(2j * np.pi * 70 * (t_z - 0.2 * t_z ** 2))
    nperseg, noverlap = (50, 40)
    win = ('gaussian', 0.01 * fs)
    f0_u, t0, Sz0_u = stft(z, fs, win, nperseg, noverlap, return_onesided=False, scaling='spectrum')
    Sz0 = fftshift(Sz0_u, axes=0)
    SFT = ShortTimeFFT.from_window(win, fs, nperseg, noverlap, fft_mode='centered', scale_to='magnitude', phase_shift=None)
    Sz1 = SFT.stft(z)
    assert_allclose(Sz0, Sz1[:, 2:-1])
    assert_allclose((abs(Sz1[:, 1]).min(), abs(Sz1[:, 1]).max()), (6.925060911593139e-07, 8.00271269218721e-07))
    t0_r, z0_r = istft(Sz0_u, fs, win, nperseg, noverlap, input_onesided=False, scaling='spectrum')
    z1_r = SFT.istft(Sz1, k1=N)
    assert len(z0_r) == N + 9
    assert_allclose(z0_r[:N], z)
    assert_allclose(z1_r, z)
    assert_allclose(SFT.spectrogram(z), abs(Sz1) ** 2)