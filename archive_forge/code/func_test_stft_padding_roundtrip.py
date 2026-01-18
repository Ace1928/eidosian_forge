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
@pytest.mark.parametrize('window, N, nperseg, noverlap, mfft', [('hann', 1024, 256, 128, 512), ('hann', 1024, 256, 128, 501), ('boxcar', 100, 10, 0, 33), (('tukey', 0.5), 1152, 256, 64, 1024), ('boxcar', 101, 10, 0, None), ('hann', 1000, 256, 128, None), ('boxcar', 100, 10, 0, None), ('boxcar', 100, 10, 9, None)])
@pytest.mark.parametrize('padding', get_args(PAD_TYPE))
def test_stft_padding_roundtrip(window, N: int, nperseg: int, noverlap: int, mfft: int, padding):
    """Test the parameter 'padding' of `stft` with roundtrips.

    The STFT parametrizations were taken from the methods
    `test_roundtrip_padded_FFT`, `test_roundtrip_padded_signal` and
    `test_roundtrip_boundary_extension` from class `TestSTFT` in  file
    ``test_spectral.py``. Note that the ShortTimeFFT does not need the
    concept of "boundary extension".
    """
    x = normal_distribution.rvs(size=N, random_state=2909)
    z = x * np.exp(1j * np.pi / 4)
    SFT = ShortTimeFFT.from_window(window, 1, nperseg, noverlap, fft_mode='twosided', mfft=mfft)
    Sx = SFT.stft(x, padding=padding)
    x1 = SFT.istft(Sx, k1=N)
    assert_allclose(x1, x, err_msg=f"Failed real roundtrip with '{padding}' padding")
    Sz = SFT.stft(z, padding=padding)
    z1 = SFT.istft(Sz, k1=N)
    assert_allclose(z1, z, err_msg='Failed complex roundtrip with ' + f" '{padding}' padding")