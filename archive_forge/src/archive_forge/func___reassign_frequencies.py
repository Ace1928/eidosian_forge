import warnings
import numpy as np
import scipy
import scipy.ndimage
import scipy.signal
import scipy.interpolate
from numba import jit
from . import convert
from .fft import get_fftlib
from .audio import resample
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..filters import get_window, semitone_filterbank
from ..filters import window_sumsquare
from numpy.typing import DTypeLike
from typing import Any, Callable, Optional, Tuple, List, Union, overload
from typing_extensions import Literal
from .._typing import _WindowSpec, _PadMode, _PadModeSTFT
def __reassign_frequencies(y: np.ndarray, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: int=2048, hop_length: Optional[int]=None, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, dtype: Optional[DTypeLike]=None, pad_mode: _PadModeSTFT='constant') -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous frequencies based on a spectrogram representation.

    The reassignment vector is calculated using equation 5.20 in Flandrin,
    Auger, & Chassande-Mottin 2002::

        omega_reassigned = omega - np.imag(S_dh/S_h)

    where ``S_h`` is the complex STFT calculated using the original window, and
    ``S_dh`` is the complex STFT calculated using the derivative of the original
    window.

    See `reassigned_spectrogram` for references.

    It is recommended to use ``pad_mode="wrap"`` or else ``center=False``, rather
    than the defaults. Frequency reassignment assumes that the energy in each
    FFT bin is associated with exactly one signal component. Reflection padding
    at the edges of the signal may invalidate the reassigned estimates in the
    boundary frames.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)], real-valued
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, t)] or None
        (optional) complex STFT calculated using the other arguments provided
        to `__reassign_frequencies`

    n_fft : int > 0 [scalar]
        FFT window size. Defaults to 2048.

    hop_length : int > 0 [scalar]
        hop length, number samples between subsequent frames.
        If not supplied, defaults to ``win_length // 4``.

    win_length : int > 0, <= n_fft
        Window length. Defaults to ``n_fft``.
        See ``stft`` for details.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a user-specified window vector of length ``n_fft``

        See `stft` for details.

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``S[:, t]`` is centered at ``y[t * hop_length]``.
        - If ``False``, then ``S[:, t]`` begins at ``y[t * hop_length]``.

    dtype : numeric type
        Complex numeric type for ``S``. Default is inferred to match
        the numerical precision of the input signal.

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    Returns
    -------
    freqs : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=real]
        Instantaneous frequencies:
        ``freqs[f, t]`` is the frequency for bin ``f``, frame ``t``.
    S : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=complex]
        Short-time Fourier transform

    Warns
    -----
    RuntimeWarning
        Frequencies with zero support will produce a divide-by-zero warning and
        will be returned as `np.nan`.

    See Also
    --------
    stft : Short-time Fourier Transform
    reassigned_spectrogram : Time-frequency reassigned spectrogram

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> frequencies, S = librosa.core.spectrum.__reassign_frequencies(y, sr=sr)
    >>> frequencies
    array([[0.000e+00, 0.000e+00, ..., 0.000e+00, 0.000e+00],
           [3.628e+00, 4.698e+00, ..., 1.239e+01, 1.072e+01],
           ...,
           [1.101e+04, 1.102e+04, ..., 1.105e+04, 1.102e+04],
           [1.102e+04, 1.102e+04, ..., 1.102e+04, 1.102e+04]])
    """
    if win_length is None:
        win_length = n_fft
    window = get_window(window, win_length, fftbins=True)
    window = util.pad_center(window, size=n_fft)
    if S is None:
        if dtype is None:
            dtype = util.dtype_r2c(y.dtype)
        S_h = stft(y=y, n_fft=n_fft, hop_length=hop_length, window=window, center=center, dtype=dtype, pad_mode=pad_mode)
    else:
        if dtype is None:
            dtype = S.dtype
        S_h = S
    window_derivative = util.cyclic_gradient(window)
    S_dh = stft(y=y, n_fft=n_fft, hop_length=hop_length, window=window_derivative, center=center, dtype=dtype, pad_mode=pad_mode)
    correction = -np.imag(S_dh / S_h)
    freqs = convert.fft_frequencies(sr=sr, n_fft=n_fft)
    freqs = util.expand_to(freqs, ndim=correction.ndim, axes=-2) + correction * (0.5 * sr / np.pi)
    return (freqs, S_h)