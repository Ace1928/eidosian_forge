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
@stencil
def _zc_stencil(x: np.ndarray, threshold: float, zero_pos: bool) -> np.ndarray:
    """Stencil to compute zero crossings"""
    x0 = x[0]
    if -threshold <= x0 <= threshold:
        x0 = 0
    x1 = x[-1]
    if -threshold <= x1 <= threshold:
        x1 = 0
    if zero_pos:
        return np.signbit(x0) != np.signbit(x1)
    else:
        return np.sign(x0) != np.sign(x1)