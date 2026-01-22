from __future__ import annotations
import copy
import datetime
import inspect
import itertools
import math
import sys
import warnings
from collections import defaultdict
from collections.abc import (
from html import escape
from numbers import Number
from operator import methodcaller
from os import PathLike
from typing import IO, TYPE_CHECKING, Any, Callable, Generic, Literal, cast, overload
import numpy as np
import pandas as pd
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftimeindex import CFTimeIndex, _parse_array_of_cftime_strings
from xarray.core import (
from xarray.core import dtypes as xrdtypes
from xarray.core._aggregations import DatasetAggregations
from xarray.core.alignment import (
from xarray.core.arithmetic import DatasetArithmetic
from xarray.core.common import (
from xarray.core.computation import unify_chunks
from xarray.core.coordinates import (
from xarray.core.duck_array_ops import datetime_to_numeric
from xarray.core.indexes import (
from xarray.core.indexing import is_fancy_indexer, map_index_queries
from xarray.core.merge import (
from xarray.core.missing import get_clean_interp_index
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import (
from xarray.core.utils import (
from xarray.core.variable import (
from xarray.namedarray.parallelcompat import get_chunked_array_type, guess_chunkmanager
from xarray.namedarray.pycompat import array_type, is_chunked_array
from xarray.plot.accessor import DatasetPlotAccessor
from xarray.util.deprecation_helpers import _deprecate_positional_args
def _stack_once(self, dims: Sequence[Hashable | ellipsis], new_dim: Hashable, index_cls: type[Index], create_index: bool | None=True) -> Self:
    if dims == ...:
        raise ValueError('Please use [...] for dims, rather than just ...')
    if ... in dims:
        dims = list(infix_dims(dims, self.dims))
    new_variables: dict[Hashable, Variable] = {}
    stacked_var_names: list[Hashable] = []
    drop_indexes: list[Hashable] = []
    for name, var in self.variables.items():
        if any((d in var.dims for d in dims)):
            add_dims = [d for d in dims if d not in var.dims]
            vdims = list(var.dims) + add_dims
            shape = [self.sizes[d] for d in vdims]
            exp_var = var.set_dims(vdims, shape)
            stacked_var = exp_var.stack(**{new_dim: dims})
            new_variables[name] = stacked_var
            stacked_var_names.append(name)
        else:
            new_variables[name] = var.copy(deep=False)
    for name in stacked_var_names:
        drop_indexes += list(self.xindexes.get_all_coords(name, errors='ignore'))
    new_indexes = {}
    new_coord_names = set(self._coord_names)
    if create_index or create_index is None:
        product_vars: dict[Any, Variable] = {}
        for dim in dims:
            idx, idx_vars = self._get_stack_index(dim, create_index=create_index)
            if idx is not None:
                product_vars.update(idx_vars)
        if len(product_vars) == len(dims):
            idx = index_cls.stack(product_vars, new_dim)
            new_indexes[new_dim] = idx
            new_indexes.update({k: idx for k in product_vars})
            idx_vars = idx.create_variables(product_vars)
            for k in idx_vars:
                new_variables.pop(k, None)
            new_variables.update(idx_vars)
            new_coord_names.update(idx_vars)
    indexes = {k: v for k, v in self._indexes.items() if k not in drop_indexes}
    indexes.update(new_indexes)
    return self._replace_with_new_dims(new_variables, coord_names=new_coord_names, indexes=indexes)