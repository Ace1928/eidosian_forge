import warnings
import numpy as np
import scipy
import numba
from .spectrum import _spectrogram
from . import convert
from .._cache import cache
from .. import util
from .. import sequence
from ..util.exceptions import ParameterError
from numpy.typing import ArrayLike
from typing import Any, Callable, Optional, Tuple, Union
from .._typing import _WindowSpec, _PadMode, _PadModeSTFT
def _parabolic_interpolation(x: np.ndarray, *, axis: int=-2) -> np.ndarray:
    """Piecewise parabolic interpolation for yin and pyin.

    Parameters
    ----------
    x : np.ndarray
        array to interpolate
    axis : int
        axis along which to interpolate

    Returns
    -------
    parabolic_shifts : np.ndarray [shape=x.shape]
        position of the parabola optima (relative to bin indices)

        Note: the shift at bin `n` is determined as 0 if the estimated
        optimum is outside the range `[n-1, n+1]`.
    """
    xi = x.swapaxes(-1, axis)
    shifts = np.empty_like(x)
    shiftsi = shifts.swapaxes(-1, axis)
    _pi_wrapper(xi, shiftsi)
    shiftsi[..., -1] = 0
    shiftsi[..., 0] = 0
    return shifts