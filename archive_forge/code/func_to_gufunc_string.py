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
def to_gufunc_string(self, exclude_dims=frozenset()):
    """Create an equivalent signature string for a NumPy gufunc.

        Unlike __str__, handles dimensions that don't map to Python
        identifiers.

        Also creates unique names for input_core_dims contained in exclude_dims.
        """
    input_core_dims = [[self.dims_map[dim] for dim in core_dims] for core_dims in self.input_core_dims]
    output_core_dims = [[self.dims_map[dim] for dim in core_dims] for core_dims in self.output_core_dims]
    if exclude_dims:
        exclude_dims = [self.dims_map[dim] for dim in exclude_dims]
        counter: Counter = Counter()

        def _enumerate(dim):
            if dim in exclude_dims:
                n = counter[dim]
                counter.update([dim])
                dim = f'{dim}_{n}'
            return dim
        input_core_dims = [[_enumerate(dim) for dim in arg] for arg in input_core_dims]
    alt_signature = type(self)(input_core_dims, output_core_dims)
    return str(alt_signature)