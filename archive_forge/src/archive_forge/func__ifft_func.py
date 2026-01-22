from __future__ import annotations
from collections.abc import Generator
from functools import cache, lru_cache, partial
from typing import Callable, get_args, Literal
import numpy as np
import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window
def _ifft_func(self, X: np.ndarray) -> np.ndarray:
    """Inverse to `_fft_func`.

        Returned is an array of length `m_num`. If the FFT is `onesided`
        then a float array is returned else a complex array is returned.
        For multidimensional arrays the transformation is carried out on the
        last axis.
        """
    if self.fft_mode == 'twosided':
        x = fft_lib.ifft(X, n=self.mfft, axis=-1)
    elif self.fft_mode == 'centered':
        x = fft_lib.ifft(fft_lib.ifftshift(X), n=self.mfft, axis=-1)
    elif self.fft_mode == 'onesided':
        x = fft_lib.irfft(X, n=self.mfft, axis=-1)
    elif self.fft_mode == 'onesided2X':
        Xc = X.copy()
        fac = np.sqrt(2) if self.scaling == 'psd' else 2
        q1 = -1 if self.mfft % 2 == 0 else None
        Xc[..., 1:q1] /= fac
        x = fft_lib.irfft(Xc, n=self.mfft, axis=-1)
    else:
        error_str = f'self.fft_mode={self.fft_mode!r} not in {get_args(FFT_MODE_TYPE)}!'
        raise RuntimeError(error_str)
    if self.phase_shift is None:
        return x[:self.m_num]
    p_s = (self.phase_shift + self.m_num_mid) % self.m_num
    return np.roll(x, p_s, axis=-1)[:self.m_num]