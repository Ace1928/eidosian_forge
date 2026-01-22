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
def apply_groupby_func(func, *args):
    """Apply a dataset or datarray level function over GroupBy, Dataset,
    DataArray, Variable and/or ndarray objects.
    """
    from xarray.core.groupby import GroupBy, peek_at
    from xarray.core.variable import Variable
    groupbys = [arg for arg in args if isinstance(arg, GroupBy)]
    assert groupbys, 'must have at least one groupby to iterate over'
    first_groupby = groupbys[0]
    grouper, = first_groupby.groupers
    if any((not grouper.group.equals(gb.groupers[0].group) for gb in groupbys[1:])):
        raise ValueError('apply_ufunc can only perform operations over multiple GroupBy objects at once if they are all grouped the same way')
    grouped_dim = grouper.name
    unique_values = grouper.unique_coord.values
    iterators = []
    for arg in args:
        iterator: Iterator[Any]
        if isinstance(arg, GroupBy):
            iterator = (value for _, value in arg)
        elif hasattr(arg, 'dims') and grouped_dim in arg.dims:
            if isinstance(arg, Variable):
                raise ValueError('groupby operations cannot be performed with xarray.Variable objects that share a dimension with the grouped dimension')
            iterator = _iter_over_selections(arg, grouped_dim, unique_values)
        else:
            iterator = itertools.repeat(arg)
        iterators.append(iterator)
    applied: Iterator = (func(*zipped_args) for zipped_args in zip(*iterators))
    applied_example, applied = peek_at(applied)
    combine = first_groupby._combine
    if isinstance(applied_example, tuple):
        combined = tuple((combine(output) for output in zip(*applied)))
    else:
        combined = combine(applied)
    return combined