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
def __check_yin_params(*, sr: float, fmax: float, fmin: float, frame_length: int, win_length: int):
    """Check the feasibility of yin/pyin parameters against
    the following conditions:

    1. 0 < fmin < fmax <= sr/2
    2. frame_length - win_length - 1 > sr/fmax
    """
    if fmax > sr / 2:
        raise ParameterError(f'fmax={fmax:.3f} cannot exceed Nyquist frequency {sr / 2}')
    if fmin >= fmax:
        raise ParameterError(f'fmin={fmin:.3f} must be less than fmax={fmax:.3f}')
    if fmin <= 0:
        raise ParameterError(f'fmin={fmin:.3f} must be strictly positive')
    if win_length >= frame_length:
        raise ParameterError(f'win_length={win_length} must be less than frame_length={frame_length}')
    if frame_length - win_length - 1 <= sr // fmax:
        fmax_feasible = sr / (frame_length - win_length - 1)
        frame_length_feasible = int(np.ceil(sr / fmax) + win_length + 1)
        raise ParameterError(f'fmax={fmax:.3f} is too small for frame_length={frame_length}, win_length={win_length}, and sr={sr}. Either increase to fmax={fmax_feasible:.3f} or frame_length={frame_length_feasible}')