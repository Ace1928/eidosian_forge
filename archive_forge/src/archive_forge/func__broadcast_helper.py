from __future__ import annotations
import functools
import operator
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Final, Generic, TypeVar, cast, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes
from xarray.core.indexes import (
from xarray.core.types import T_Alignable
from xarray.core.utils import is_dict_like, is_full_slice
from xarray.core.variable import Variable, as_compatible_data, calculate_dimensions
def _broadcast_helper(arg: T_Alignable, exclude, dims_map, common_coords) -> T_Alignable:
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset

    def _set_dims(var):
        var_dims_map = dims_map.copy()
        for dim in exclude:
            with suppress(ValueError):
                var_dims_map[dim] = var.shape[var.dims.index(dim)]
        return var.set_dims(var_dims_map)

    def _broadcast_array(array: T_DataArray) -> T_DataArray:
        data = _set_dims(array.variable)
        coords = dict(array.coords)
        coords.update(common_coords)
        return array.__class__(data, coords, data.dims, name=array.name, attrs=array.attrs)

    def _broadcast_dataset(ds: T_Dataset) -> T_Dataset:
        data_vars = {k: _set_dims(ds.variables[k]) for k in ds.data_vars}
        coords = dict(ds.coords)
        coords.update(common_coords)
        return ds.__class__(data_vars, coords, ds.attrs)
    if isinstance(arg, DataArray):
        return cast(T_Alignable, _broadcast_array(arg))
    elif isinstance(arg, Dataset):
        return cast(T_Alignable, _broadcast_dataset(arg))
    else:
        raise ValueError('all input must be Dataset or DataArray objects')