import functools
import warnings
from collections.abc import Hashable
from typing import Union
import numpy as np
import pandas as pd
from xarray.core import duck_array_ops, formatting, utils
from xarray.core.coordinates import Coordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import Index, PandasIndex, PandasMultiIndex, default_indexes
from xarray.core.variable import IndexVariable, Variable
def _assert_internal_invariants(xarray_obj: Union[DataArray, Dataset, Variable], check_default_indexes: bool):
    """Validate that an xarray object satisfies its own internal invariants.

    This exists for the benefit of xarray's own test suite, but may be useful
    in external projects if they (ill-advisedly) create objects using xarray's
    private APIs.
    """
    if isinstance(xarray_obj, Variable):
        _assert_variable_invariants(xarray_obj)
    elif isinstance(xarray_obj, DataArray):
        _assert_dataarray_invariants(xarray_obj, check_default_indexes=check_default_indexes)
    elif isinstance(xarray_obj, Dataset):
        _assert_dataset_invariants(xarray_obj, check_default_indexes=check_default_indexes)
    elif isinstance(xarray_obj, Coordinates):
        _assert_dataset_invariants(xarray_obj.to_dataset(), check_default_indexes=check_default_indexes)
    else:
        raise TypeError(f'{type(xarray_obj)} is not a supported type for xarray invariant checks')