from __future__ import annotations
import os
import pathlib
import warnings
import soundfile as sf
import audioread
import numpy as np
import scipy.signal
import soxr
import lazy_loader as lazy
from numba import jit, stencil, guvectorize
from .fft import get_fftlib
from .convert import frames_to_samples, time_to_samples
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..util.decorators import deprecated
from ..util.deprecation import Deprecated, rename_kw
from .._typing import _FloatLike_co, _IntLike_co, _SequenceLike
from typing import Any, BinaryIO, Callable, Generator, Optional, Tuple, Union, List
from numpy.typing import DTypeLike, ArrayLike
@guvectorize(['void(float32[:], float32, bool_, bool_[:])', 'void(float64[:], float64, bool_, bool_[:])'], '(n),(),()->(n)', cache=True, nopython=True)
def _zc_wrapper(x: np.ndarray, threshold: float, zero_pos: bool, y: np.ndarray) -> None:
    """Vectorized wrapper for zero crossing stencil"""
    y[:] = _zc_stencil(x, threshold, zero_pos)