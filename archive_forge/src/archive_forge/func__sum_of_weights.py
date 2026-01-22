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
def _sum_of_weights(self, da: T_DataArray, dim: Dims=None) -> T_DataArray:
    """Calculate the sum of weights, accounting for missing values"""
    mask = da.notnull()
    if self.weights.dtype == bool:
        sum_of_weights = self._reduce(mask, duck_array_ops.astype(self.weights, dtype=int), dim=dim, skipna=False)
    else:
        sum_of_weights = self._reduce(mask, self.weights, dim=dim, skipna=False)
    valid_weights = sum_of_weights != 0.0
    return sum_of_weights.where(valid_weights)