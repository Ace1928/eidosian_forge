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
@pytest.mark.parametrize('window, n, nperseg, noverlap', [('boxcar', 100, 10, 0), ('boxcar', 100, 10, 9), ('bartlett', 101, 51, 26), ('hann', 1024, 256, 128), (('tukey', 0.5), 1152, 256, 64), ('hann', 1024, 256, 255), ('boxcar', 100, 10, 3), ('bartlett', 101, 51, 37), ('hann', 1024, 256, 127), (('tukey', 0.5), 1152, 256, 14), ('hann', 1024, 256, 5)])
def test_roundtrip_windows(window, n: int, nperseg: int, noverlap: int):
    """Roundtrip test adapted from `test_spectral.TestSTFT`.

    The parameters are taken from the methods test_roundtrip_real(),
    test_roundtrip_nola_not_cola(), test_roundtrip_float32(),
    test_roundtrip_complex().
    """
    np.random.seed(2394655)
    w = get_window(window, nperseg)
    SFT = ShortTimeFFT(w, nperseg - noverlap, fs=1, fft_mode='twosided', phase_shift=None)
    z = 10 * np.random.randn(n) + 10j * np.random.randn(n)
    Sz = SFT.stft(z)
    z1 = SFT.istft(Sz, k1=len(z))
    assert_allclose(z, z1, err_msg='Roundtrip for complex values failed')
    x = 10 * np.random.randn(n)
    Sx = SFT.stft(x)
    x1 = SFT.istft(Sx, k1=len(z))
    assert_allclose(x, x1, err_msg='Roundtrip for float values failed')
    x32 = x.astype(np.float32)
    Sx32 = SFT.stft(x32)
    x32_1 = SFT.istft(Sx32, k1=len(x32))
    assert_allclose(x32, x32_1, err_msg='Roundtrip for 32 Bit float values failed')