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
@numba.jit(nopython=True, cache=True)
def __is_unique(x):
    """Determine if the input array has all unique values.

    This function is a helper for `is_unique` and is not
    to be called directly.
    """
    uniques = np.unique(x)
    return uniques.shape[0] == x.size