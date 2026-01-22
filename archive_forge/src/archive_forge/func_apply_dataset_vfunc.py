from __future__ import annotations
import functools
import itertools
import operator
import warnings
from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence, Set
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, Union, cast, overload
import numpy as np
from xarray.core import dtypes, duck_array_ops, utils
from xarray.core.alignment import align, deep_align
from xarray.core.common import zeros_like
from xarray.core.duck_array_ops import datetime_to_numeric
from xarray.core.formatting import limit_lines
from xarray.core.indexes import Index, filter_indexes_from_coords
from xarray.core.merge import merge_attrs, merge_coordinates_without_align
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import Dims, T_DataArray
from xarray.core.utils import is_dict_like, is_duck_dask_array, is_scalar, parse_dims
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
from xarray.util.deprecation_helpers import deprecate_dims
def apply_dataset_vfunc(func, *args, signature: _UFuncSignature, join='inner', dataset_join='exact', fill_value=_NO_FILL_VALUE, exclude_dims=frozenset(), keep_attrs='override', on_missing_core_dim: MissingCoreDimOptions='raise') -> Dataset | tuple[Dataset, ...]:
    """Apply a variable level function over Dataset, dict of DataArray,
    DataArray, Variable and/or ndarray objects.
    """
    from xarray.core.dataset import Dataset
    if dataset_join not in _JOINS_WITHOUT_FILL_VALUES and fill_value is _NO_FILL_VALUE:
        raise TypeError('to apply an operation to datasets with different data variables with apply_ufunc, you must supply the dataset_fill_value argument.')
    objs = _all_of_type(args, Dataset)
    if len(args) > 1:
        args = tuple(deep_align(args, join=join, copy=False, exclude=exclude_dims, raise_on_invalid=False))
    list_of_coords, list_of_indexes = build_output_coords_and_indexes(args, signature, exclude_dims, combine_attrs=keep_attrs)
    args = tuple((getattr(arg, 'data_vars', arg) for arg in args))
    result_vars = apply_dict_of_variables_vfunc(func, *args, signature=signature, join=dataset_join, fill_value=fill_value, on_missing_core_dim=on_missing_core_dim)
    out: Dataset | tuple[Dataset, ...]
    if signature.num_outputs > 1:
        out = tuple((_fast_dataset(*args) for args in zip(result_vars, list_of_coords, list_of_indexes)))
    else:
        coord_vars, = list_of_coords
        indexes, = list_of_indexes
        out = _fast_dataset(result_vars, coord_vars, indexes=indexes)
    attrs = merge_attrs([x.attrs for x in objs], combine_attrs=keep_attrs)
    if isinstance(out, tuple):
        for ds in out:
            ds.attrs = attrs
    else:
        out.attrs = attrs
    return out