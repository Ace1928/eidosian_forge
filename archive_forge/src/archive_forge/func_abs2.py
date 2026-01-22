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
def abs2(x: _NumberOrArray, dtype: Optional[DTypeLike]=None) -> _NumberOrArray:
    """Compute the squared magnitude of a real or complex array.

    This function is equivalent to calling `np.abs(x)**2` but it
    is slightly more efficient.

    Parameters
    ----------
    x : np.ndarray or scalar, real or complex typed
        The input data, either real (float32, float64) or complex (complex64, complex128) typed
    dtype : np.dtype, optional
        The data type of the output array.
        If not provided, it will be inferred from `x`

    Returns
    -------
    p : np.ndarray or scale, real
        squared magnitude of `x`

    Examples
    --------
    >>> librosa.util.abs2(3 + 4j)
    25.0

    >>> librosa.util.abs2((0.5j)**np.arange(8))
    array([1.000e+00, 2.500e-01, 6.250e-02, 1.562e-02, 3.906e-03, 9.766e-04,
       2.441e-04, 6.104e-05])
    """
    if np.iscomplexobj(x):
        y = _cabs2(x)
        if dtype is None:
            return y
        else:
            return y.astype(dtype)
    else:
        return np.power(x, 2, dtype=dtype)