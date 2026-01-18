from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def times_like(X: Union[np.ndarray, float], *, sr: float=22050, hop_length: int=512, n_fft: Optional[int]=None, axis: int=-1) -> np.ndarray:
    """Return an array of time values to match the time axis from a feature matrix.

    Parameters
    ----------
    X : np.ndarray or scalar
        - If ndarray, X is a feature matrix, e.g. STFT, chromagram, or mel spectrogram.
        - If scalar, X represents the number of frames.
    sr : number > 0 [scalar]
        audio sampling rate
    hop_length : int > 0 [scalar]
        number of samples between successive frames
    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``n_fft // 2``
        to counteract windowing effects when using a non-centered STFT.
    axis : int [scalar]
        The axis representing the time axis of X.
        By default, the last axis (-1) is taken.

    Returns
    -------
    times : np.ndarray [shape=(n,)]
        ndarray of times (in seconds) corresponding to each frame of X.

    See Also
    --------
    samples_like :
        Return an array of sample indices to match the time axis from a feature matrix.

    Examples
    --------
    Provide a feature matrix input:

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> D = librosa.stft(y)
    >>> times = librosa.times_like(D)
    >>> times
    array([0.   , 0.023, ..., 5.294, 5.317])

    Provide a scalar input:

    >>> n_frames = 2647
    >>> times = librosa.times_like(n_frames)
    >>> times
    array([  0.00000000e+00,   2.32199546e-02,   4.64399093e-02, ...,
             6.13935601e+01,   6.14167800e+01,   6.14400000e+01])
    """
    samples = samples_like(X, hop_length=hop_length, n_fft=n_fft, axis=axis)
    time: np.ndarray = samples_to_time(samples, sr=sr)
    return time