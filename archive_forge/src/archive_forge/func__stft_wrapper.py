import platform
from typing import cast, Literal
import numpy as np
from numpy.testing import assert_allclose
from scipy.signal import ShortTimeFFT
from scipy.signal import csd, get_window, stft, istft
from scipy.signal._arraytools import const_ext, even_ext, odd_ext, zero_ext
from scipy.signal._short_time_fft import FFT_MODE_TYPE
from scipy.signal._spectral_py import _spectral_helper, _triage_segments, \
def _stft_wrapper(x, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1, scaling='spectrum'):
    """Wrapper for the SciPy `stft()` function based on `ShortTimeFFT` for
    unit testing.

    Handling the boundary and padding is where `ShortTimeFFT` and `stft()`
    differ in behavior. Parts of `_spectral_helper()` were copied to mimic
    the` stft()` behavior.

    This function is meant to be solely used by `stft_compare()`.
    """
    if scaling not in ('psd', 'spectrum'):
        raise ValueError(f"Parameter scaling={scaling!r} not in ['spectrum', 'psd']!")
    boundary_funcs = {'even': even_ext, 'odd': odd_ext, 'constant': const_ext, 'zeros': zero_ext, None: None}
    if boundary not in boundary_funcs:
        raise ValueError(f"Unknown boundary option '{boundary}', must be one" + f' of: {list(boundary_funcs.keys())}')
    if x.size == 0:
        return (np.empty(x.shape), np.empty(x.shape), np.empty(x.shape))
    if nperseg is not None:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')
    win, nperseg = _triage_segments(window, nperseg, input_length=x.shape[axis])
    if nfft is None:
        nfft = nperseg
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
    n = x.shape[axis]
    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, nperseg // 2, axis=axis)
    if padded:
        x = np.moveaxis(x, axis, -1)
        if n % 2 == 1 and nperseg % 2 == 1 and (noverlap % 2 == 1):
            x = x[..., :axis - 1]
        nadd = -(x.shape[-1] - nperseg) % nstep % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = np.concatenate((x, np.zeros(zeros_shape)), axis=-1)
        x = np.moveaxis(x, -1, axis)
    scale_to = {'spectrum': 'magnitude', 'psd': 'psd'}[scaling]
    if np.iscomplexobj(x) and return_onesided:
        return_onesided = False
    fft_mode = cast(FFT_MODE_TYPE, 'onesided' if return_onesided else 'twosided')
    ST = ShortTimeFFT(win, nstep, fs, fft_mode=fft_mode, mfft=nfft, scale_to=scale_to, phase_shift=None)
    k_off = nperseg // 2
    p0 = 0
    nn = x.shape[axis] if padded else n + k_off + 1
    p1 = ST.upper_border_begin(nn)[1]
    if padded is True and nperseg - noverlap == 1:
        p1 -= nperseg // 2 - 1
    detr = None if detrend is False else detrend
    Sxx = ST.stft_detrend(x, detr, p0, p1, k_offset=k_off, axis=axis)
    t = ST.t(nn, 0, p1 - p0, k_offset=0 if boundary is not None else k_off)
    if x.dtype in (np.float32, np.complex64):
        Sxx = Sxx.astype(np.complex64)
    if boundary is None and padded is False:
        t, Sxx = (t[1:-1], Sxx[..., :-2])
        t -= k_off / fs
    return (ST.f, t, Sxx)