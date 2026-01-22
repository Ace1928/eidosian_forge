import platform
from typing import cast, Literal
import numpy as np
from numpy.testing import assert_allclose
from scipy.signal import ShortTimeFFT
from scipy.signal import csd, get_window, stft, istft
from scipy.signal._arraytools import const_ext, even_ext, odd_ext, zero_ext
from scipy.signal._short_time_fft import FFT_MODE_TYPE
from scipy.signal._spectral_py import _spectral_helper, _triage_segments, \
def _csd_test_shim(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1):
    """Compare output of  _spectral_helper() and ShortTimeFFT, more
    precisely _spect_helper_csd() for used in csd_wrapper().

   The motivation of this function is to test if the ShortTimeFFT-based
   wrapper `_spect_helper_csd()` returns the same values as `_spectral_helper`.
   This function should only be usd by csd() in (unit) testing.
   """
    freqs, t, Pxy = _spectral_helper(x, y, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling, axis, mode='psd')
    freqs1, Pxy1 = _spect_helper_csd(x, y, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling, axis)
    np.testing.assert_allclose(freqs1, freqs)
    amax_Pxy = max(np.abs(Pxy).max(), 1) if Pxy.size else 1
    atol = np.finfo(Pxy.dtype).resolution * amax_Pxy
    np.testing.assert_allclose(Pxy1, Pxy, atol=atol)
    return (freqs, t, Pxy)