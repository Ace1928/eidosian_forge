import platform
from typing import cast, Literal
import numpy as np
from numpy.testing import assert_allclose
from scipy.signal import ShortTimeFFT
from scipy.signal import csd, get_window, stft, istft
from scipy.signal._arraytools import const_ext, even_ext, odd_ext, zero_ext
from scipy.signal._short_time_fft import FFT_MODE_TYPE
from scipy.signal._spectral_py import _spectral_helper, _triage_segments, \
def _spect_helper_csd(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1):
    """Wrapper for replacing _spectral_helper() by using the ShortTimeFFT
      for use by csd().

    This function should be only used by _csd_test_shim() and is only useful
    for testing the ShortTimeFFT implementation.
    """
    same_data = y is x
    axis = int(axis)
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)
    if not same_data:
        xouter = list(x.shape)
        youter = list(y.shape)
        xouter.pop(axis)
        youter.pop(axis)
        try:
            outershape = np.broadcast(np.empty(xouter), np.empty(youter)).shape
        except ValueError as e:
            raise ValueError('x and y cannot be broadcast together.') from e
    if same_data:
        if x.size == 0:
            return (np.empty(x.shape), np.empty(x.shape))
    elif x.size == 0 or y.size == 0:
        outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
        emptyout = np.moveaxis(np.empty(outshape), -1, axis)
        return (emptyout, emptyout)
    if nperseg is not None:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')
    n = x.shape[axis] if same_data else max(x.shape[axis], y.shape[axis])
    win, nperseg = _triage_segments(window, nperseg, input_length=n)
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
    if np.iscomplexobj(x) and return_onesided:
        return_onesided = False
    fft_mode = cast(FFT_MODE_TYPE, 'onesided' if return_onesided else 'twosided')
    scale = {'spectrum': 'magnitude', 'density': 'psd'}[scaling]
    SFT = ShortTimeFFT(win, nstep, fs, fft_mode=fft_mode, mfft=nfft, scale_to=scale, phase_shift=None)
    Pxy = SFT.spectrogram(y, x, detr=None if detrend is False else detrend, p0=0, p1=(n - noverlap) // SFT.hop, k_offset=nperseg // 2, axis=axis).conj()
    if return_onesided:
        f_axis = Pxy.ndim - 1 + axis if axis < 0 else axis
        Pxy = np.moveaxis(Pxy, f_axis, -1)
        Pxy[..., 1:-1 if SFT.mfft % 2 == 0 else None] *= 2
        Pxy = np.moveaxis(Pxy, -1, f_axis)
    return (SFT.f, Pxy)