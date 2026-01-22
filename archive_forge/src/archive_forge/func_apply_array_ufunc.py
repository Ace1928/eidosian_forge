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
def apply_array_ufunc(func, *args, dask='forbidden'):
    """Apply a ndarray level function over ndarray objects."""
    if any((is_chunked_array(arg) for arg in args)):
        if dask == 'forbidden':
            raise ValueError('apply_ufunc encountered a dask array on an argument, but handling for dask arrays has not been enabled. Either set the ``dask`` argument or load your data into memory first with ``.load()`` or ``.compute()``')
        elif dask == 'parallelized':
            raise ValueError("cannot use dask='parallelized' for apply_ufunc unless at least one input is an xarray object")
        elif dask == 'allowed':
            pass
        else:
            raise ValueError(f'unknown setting for dask array handling: {dask}')
    return func(*args)