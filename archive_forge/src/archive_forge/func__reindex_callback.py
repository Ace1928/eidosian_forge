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
def _reindex_callback(self, aligner: alignment.Aligner, dim_pos_indexers: dict[Hashable, Any], variables: dict[Hashable, Variable], indexes: dict[Hashable, Index], fill_value: Any, exclude_dims: frozenset[Hashable], exclude_vars: frozenset[Hashable]) -> Self:
    """Callback called from ``Aligner`` to create a new reindexed Dataset."""
    new_variables = variables.copy()
    new_indexes = indexes.copy()
    for name, new_var in new_variables.items():
        var = self._variables.get(name)
        if var is not None:
            new_var.attrs = var.attrs
            new_var.encoding = var.encoding
    for name, idx in self._indexes.items():
        var = self._variables[name]
        if set(var.dims) <= exclude_dims:
            new_indexes[name] = idx
            new_variables[name] = var
    if not dim_pos_indexers:
        if set(new_indexes) - set(self._indexes):
            reindexed = self._overwrite_indexes(new_indexes, new_variables)
        else:
            reindexed = self.copy(deep=aligner.copy)
    else:
        to_reindex = {k: v for k, v in self.variables.items() if k not in variables and k not in exclude_vars}
        reindexed_vars = alignment.reindex_variables(to_reindex, dim_pos_indexers, copy=aligner.copy, fill_value=fill_value, sparse=aligner.sparse)
        new_variables.update(reindexed_vars)
        new_coord_names = self._coord_names | set(new_indexes)
        reindexed = self._replace_with_new_dims(new_variables, new_coord_names, indexes=new_indexes)
    reindexed.encoding = self.encoding
    return reindexed