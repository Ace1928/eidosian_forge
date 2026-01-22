from __future__ import annotations
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any, Union, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes, utils
from xarray.core.alignment import align, reindex_variables
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.core.indexes import Index, PandasIndex
from xarray.core.merge import (
from xarray.core.types import T_DataArray, T_Dataset, T_Variable
from xarray.core.variable import Variable
from xarray.core.variable import concat as concat_vars
def _dataarray_concat(arrays: Iterable[T_DataArray], dim: str | T_Variable | T_DataArray | pd.Index, data_vars: T_DataVars, coords: str | list[str], compat: CompatOptions, positions: Iterable[Iterable[int]] | None, fill_value: object=dtypes.NA, join: JoinOptions='outer', combine_attrs: CombineAttrsOptions='override') -> T_DataArray:
    from xarray.core.dataarray import DataArray
    arrays = list(arrays)
    if not all((isinstance(array, DataArray) for array in arrays)):
        raise TypeError("The elements in the input list need to be either all 'Dataset's or all 'DataArray's")
    if data_vars != 'all':
        raise ValueError('data_vars is not a valid argument when concatenating DataArray objects')
    datasets = []
    for n, arr in enumerate(arrays):
        if n == 0:
            name = arr.name
        elif name != arr.name:
            if compat == 'identical':
                raise ValueError('array names not identical')
            else:
                arr = arr.rename(name)
        datasets.append(arr._to_temp_dataset())
    ds = _dataset_concat(datasets, dim, data_vars, coords, compat, positions, fill_value=fill_value, join=join, combine_attrs=combine_attrs)
    merged_attrs = merge_attrs([da.attrs for da in arrays], combine_attrs)
    result = arrays[0]._from_temp_dataset(ds, name)
    result.attrs = merged_attrs
    return result