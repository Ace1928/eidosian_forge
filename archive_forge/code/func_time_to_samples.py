from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def time_to_samples(times: _ScalarOrSequence[_FloatLike_co], *, sr: float=22050) -> Union[np.integer[Any], np.ndarray]:
    """Convert timestamps (in seconds) to sample indices.

    Parameters
    ----------
    times : number or np.ndarray
        Time value or array of time values (in seconds)
    sr : number > 0
        Sampling rate

    Returns
    -------
    samples : int or np.ndarray [shape=times.shape, dtype=int]
        Sample indices corresponding to values in ``times``

    See Also
    --------
    time_to_frames : convert time values to frame indices
    samples_to_time : convert sample indices to time values

    Examples
    --------
    >>> librosa.time_to_samples(np.arange(0, 1, 0.1), sr=22050)
    array([    0,  2205,  4410,  6615,  8820, 11025, 13230, 15435,
           17640, 19845])
    """
    return (np.asanyarray(times) * sr).astype(int)