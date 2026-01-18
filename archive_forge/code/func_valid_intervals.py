from __future__ import annotations
import scipy.ndimage
import scipy.sparse
import numpy as np
import numba
from numpy.lib.stride_tricks import as_strided
from .._cache import cache
from .exceptions import ParameterError
from .deprecation import Deprecated
from numpy.typing import ArrayLike, DTypeLike
from typing import (
from typing_extensions import Literal
from .._typing import _SequenceLike, _FloatLike_co, _ComplexLike_co
def valid_intervals(intervals: np.ndarray) -> bool:
    """Ensure that an array is a valid representation of time intervals:

        - intervals.ndim == 2
        - intervals.shape[1] == 2
        - intervals[i, 0] <= intervals[i, 1] for all i

    Parameters
    ----------
    intervals : np.ndarray [shape=(n, 2)]
        set of time intervals

    Returns
    -------
    valid : bool
        True if ``intervals`` passes validation.
    """
    if intervals.ndim != 2 or intervals.shape[-1] != 2:
        raise ParameterError('intervals must have shape (n, 2)')
    if np.any(intervals[:, 0] > intervals[:, 1]):
        raise ParameterError(f'intervals={intervals} must have non-negative durations')
    return True