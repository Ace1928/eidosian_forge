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
def _copy_listed(self, names: Iterable[Hashable]) -> Self:
    """Create a new Dataset with the listed variables from this dataset and
        the all relevant coordinates. Skips all validation.
        """
    variables: dict[Hashable, Variable] = {}
    coord_names = set()
    indexes: dict[Hashable, Index] = {}
    for name in names:
        try:
            variables[name] = self._variables[name]
        except KeyError:
            ref_name, var_name, var = _get_virtual_variable(self._variables, name, self.sizes)
            variables[var_name] = var
            if ref_name in self._coord_names or ref_name in self.dims:
                coord_names.add(var_name)
            if (var_name,) == var.dims:
                index, index_vars = create_default_index_implicit(var, names)
                indexes.update({k: index for k in index_vars})
                variables.update(index_vars)
                coord_names.update(index_vars)
    needed_dims: OrderedSet[Hashable] = OrderedSet()
    for v in variables.values():
        needed_dims.update(v.dims)
    dims = {k: self.sizes[k] for k in needed_dims}
    for k in self._variables:
        if k not in self._coord_names:
            continue
        if set(self.variables[k].dims) <= needed_dims:
            variables[k] = self._variables[k]
            coord_names.add(k)
    indexes.update(filter_indexes_from_coords(self._indexes, coord_names))
    return self._replace(variables, coord_names, dims, indexes=indexes)