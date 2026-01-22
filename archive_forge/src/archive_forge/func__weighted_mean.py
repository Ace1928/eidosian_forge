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
def _weighted_mean(self, da: T_DataArray, dim: Dims=None, skipna: bool | None=None) -> T_DataArray:
    """Reduce a DataArray by a weighted ``mean`` along some dimension(s)."""
    weighted_sum = self._weighted_sum(da, dim=dim, skipna=skipna)
    sum_of_weights = self._sum_of_weights(da, dim=dim)
    return weighted_sum / sum_of_weights