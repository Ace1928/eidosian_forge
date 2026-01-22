import warnings
import numpy as np
from numba import jit
from . import audio
from .intervals import interval_frequencies
from .fft import get_fftlib
from .convert import cqt_frequencies, note_to_hz
from .spectrum import stft, istft
from .pitch import estimate_tuning
from .._cache import cache
from .. import filters
from .. import util
from ..util.exceptions import ParameterError
from numpy.typing import DTypeLike
from typing import Optional, Union, Collection, List
from .._typing import _WindowSpec, _PadMode, _FloatLike_co, _ensure_not_reachable
def __et_relative_bw(bins_per_octave: int) -> np.ndarray:
    """Compute the relative bandwidth coefficient for equal
    (geometric) freuqency spacing and a give number of bins
    per octave.

    This is a special case of the more general `relative_bandwidth`
    calculation that can be used when only a single basis frequency
    is used.

    Parameters
    ----------
    bins_per_octave : int

    Returns
    -------
    alpha : np.ndarray > 0
        Value is cast up to a 1d array to allow slicing
    """
    r = 2 ** (1 / bins_per_octave)
    return np.atleast_1d((r ** 2 - 1) / (r ** 2 + 1))