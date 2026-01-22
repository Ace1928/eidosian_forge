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
def __early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves):
    """Compute the number of early downsampling operations"""
    downsample_count1 = max(0, int(np.ceil(np.log2(nyquist / filter_cutoff)) - 1) - 1)
    num_twos = __num_two_factors(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)
    return min(downsample_count1, downsample_count2)