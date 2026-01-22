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
@numba.vectorize(['float32(complex64)', 'float64(complex128)'], nopython=True, cache=True, identity=0)
def _cabs2(x: _ComplexLike_co) -> _FloatLike_co:
    """Efficiently compute abs2 on complex inputs"""
    return x.real ** 2 + x.imag ** 2