import platform
from typing import cast, Literal
import numpy as np
from numpy.testing import assert_allclose
from scipy.signal import ShortTimeFFT
from scipy.signal import csd, get_window, stft, istft
from scipy.signal._arraytools import const_ext, even_ext, odd_ext, zero_ext
from scipy.signal._short_time_fft import FFT_MODE_TYPE
from scipy.signal._spectral_py import _spectral_helper, _triage_segments, \
def _istft_wrapper(Zxx, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2, scaling='spectrum') -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Wrapper for the SciPy `istft()` function based on `ShortTimeFFT` for
        unit testing.

    Note that only option handling is implemented as far as to handle the unit
    tests. E.g., the case ``nperseg=None`` is not handled.

    This function is meant to be solely used by `istft_compare()`.
    """
    if Zxx.ndim < 2:
        raise ValueError('Input stft must be at least 2d!')
    if freq_axis == time_axis:
        raise ValueError('Must specify differing time and frequency axes!')
    nseg = Zxx.shape[time_axis]
    if input_onesided:
        n_default = 2 * (Zxx.shape[freq_axis] - 1)
    else:
        n_default = Zxx.shape[freq_axis]
    if nperseg is None:
        nperseg = n_default
    else:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')
    if nfft is None:
        if input_onesided and nperseg == n_default + 1:
            nfft = nperseg
        else:
            nfft = n_default
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)
    if noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap
    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError(f'window must have length of {nperseg}')
    outputlength = nperseg + (nseg - 1) * nstep
    fft_mode = cast(FFT_MODE_TYPE, 'onesided' if input_onesided else 'twosided')
    scale_to = cast(Literal['magnitude', 'psd'], {'spectrum': 'magnitude', 'psd': 'psd'}[scaling])
    ST = ShortTimeFFT(win, nstep, fs, fft_mode=fft_mode, mfft=nfft, scale_to=scale_to, phase_shift=None)
    if boundary:
        j = nperseg if nperseg % 2 == 0 else nperseg - 1
        k0 = ST.k_min + nperseg // 2
        k1 = outputlength - j + k0
    else:
        raise NotImplementedError('boundary=False does not make sense with' + 'ShortTimeFFT.istft()!')
    x = ST.istft(Zxx, k0=k0, k1=k1, f_axis=freq_axis, t_axis=time_axis)
    t = np.arange(k1 - k0) * ST.T
    k_hi = ST.upper_border_begin(k1 - k0)[0]
    return (t, x, (ST.lower_border_end[0], k_hi))