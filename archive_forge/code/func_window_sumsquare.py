import warnings
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
from numba import jit
from ._cache import cache
from . import util
from .util.exceptions import ParameterError
from .util.decorators import deprecated
from .core.convert import note_to_hz, hz_to_midi, midi_to_hz, hz_to_octs
from .core.convert import fft_frequencies, mel_frequencies
from numpy.typing import ArrayLike, DTypeLike
from typing import Any, List, Optional, Tuple, Union
from typing_extensions import Literal
from ._typing import _WindowSpec, _FloatLike_co
def window_sumsquare(*, window: _WindowSpec, n_frames: int, hop_length: int=512, win_length: Optional[int]=None, n_fft: int=2048, dtype: DTypeLike=np.float32, norm: Optional[float]=None) -> np.ndarray:
    """Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing observations
    in short-time Fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches ``n_fft``.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    norm : {np.inf, -np.inf, 0, float > 0, None}
        Normalization mode used in window construction.
        Note that this does not affect the squaring operation.

    Returns
    -------
    wss : np.ndarray, shape=``(n_fft + hop_length * (n_frames - 1))``
        The sum-squared envelope of the window function

    Examples
    --------
    For a fixed frame length (2048), compare modulation effects for a Hann window
    at different hop lengths:

    >>> n_frames = 50
    >>> wss_256 = librosa.filters.window_sumsquare(window='hann', n_frames=n_frames, hop_length=256)
    >>> wss_512 = librosa.filters.window_sumsquare(window='hann', n_frames=n_frames, hop_length=512)
    >>> wss_1024 = librosa.filters.window_sumsquare(window='hann', n_frames=n_frames, hop_length=1024)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharey=True)
    >>> ax[0].plot(wss_256)
    >>> ax[0].set(title='hop_length=256')
    >>> ax[1].plot(wss_512)
    >>> ax[1].set(title='hop_length=512')
    >>> ax[2].plot(wss_1024)
    >>> ax[2].set(title='hop_length=1024')
    """
    if win_length is None:
        win_length = n_fft
    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)
    win_sq = get_window(window, win_length)
    win_sq = util.normalize(win_sq, norm=norm) ** 2
    win_sq = util.pad_center(win_sq, size=n_fft)
    __window_ss_fill(x, win_sq, n_frames, hop_length)
    return x