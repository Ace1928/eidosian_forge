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
def _overwrite_indexes(self, indexes: Mapping[Hashable, Index], variables: Mapping[Hashable, Variable] | None=None, drop_variables: list[Hashable] | None=None, drop_indexes: list[Hashable] | None=None, rename_dims: Mapping[Hashable, Hashable] | None=None) -> Self:
    """Maybe replace indexes.

        This function may do a lot more depending on index query
        results.

        """
    if not indexes:
        return self
    if variables is None:
        variables = {}
    if drop_variables is None:
        drop_variables = []
    if drop_indexes is None:
        drop_indexes = []
    new_variables = self._variables.copy()
    new_coord_names = self._coord_names.copy()
    new_indexes = dict(self._indexes)
    index_variables = {}
    no_index_variables = {}
    for name, var in variables.items():
        old_var = self._variables.get(name)
        if old_var is not None:
            var.attrs.update(old_var.attrs)
            var.encoding.update(old_var.encoding)
        if name in indexes:
            index_variables[name] = var
        else:
            no_index_variables[name] = var
    for name in indexes:
        new_indexes[name] = indexes[name]
    for name, var in index_variables.items():
        new_coord_names.add(name)
        new_variables[name] = var
    for k in no_index_variables:
        new_variables.pop(k)
    new_variables.update(no_index_variables)
    for name in drop_indexes:
        new_indexes.pop(name)
    for name in drop_variables:
        new_variables.pop(name)
        new_indexes.pop(name, None)
        new_coord_names.remove(name)
    replaced = self._replace(variables=new_variables, coord_names=new_coord_names, indexes=new_indexes)
    if rename_dims:
        dims = replaced._rename_dims(rename_dims)
        new_variables, new_coord_names = replaced._rename_vars({}, rename_dims)
        return replaced._replace(variables=new_variables, coord_names=new_coord_names, dims=dims)
    else:
        return replaced