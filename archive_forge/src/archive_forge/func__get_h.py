from __future__ import annotations
from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Generic, Literal, cast
import numpy as np
from numpy.typing import ArrayLike
from xarray.core import duck_array_ops, utils
from xarray.core.alignment import align, broadcast
from xarray.core.computation import apply_ufunc, dot
from xarray.core.types import Dims, T_DataArray, T_Xarray
from xarray.namedarray.utils import is_duck_dask_array
from xarray.util.deprecation_helpers import _deprecate_positional_args
def _get_h(n: float, q: np.ndarray, method: QUANTILE_METHODS) -> np.ndarray:
    """Return the interpolation parameter."""
    h: np.ndarray
    if method == 'linear':
        h = (n - 1) * q + 1
    elif method == 'interpolated_inverted_cdf':
        h = n * q
    elif method == 'hazen':
        h = n * q + 0.5
    elif method == 'weibull':
        h = (n + 1) * q
    elif method == 'median_unbiased':
        h = (n + 1 / 3) * q + 1 / 3
    elif method == 'normal_unbiased':
        h = (n + 1 / 4) * q + 3 / 8
    else:
        raise ValueError(f'Invalid method: {method}.')
    return h.clip(1, n)