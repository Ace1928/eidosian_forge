from __future__ import annotations
import datetime
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from os import PathLike
from typing import (
import numpy as np
import pandas as pd
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import alignment, computation, dtypes, indexing, ops, utils
from xarray.core._aggregations import DataArrayAggregations
from xarray.core.accessor_dt import CombinedDatetimelikeAccessor
from xarray.core.accessor_str import StringAccessor
from xarray.core.alignment import (
from xarray.core.arithmetic import DataArrayArithmetic
from xarray.core.common import AbstractArray, DataWithCoords, get_chunksizes
from xarray.core.computation import unify_chunks
from xarray.core.coordinates import (
from xarray.core.dataset import Dataset
from xarray.core.formatting import format_item
from xarray.core.indexes import (
from xarray.core.indexing import is_fancy_indexer, map_index_queries
from xarray.core.merge import PANDAS_TYPES, MergeError
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import (
from xarray.core.utils import (
from xarray.core.variable import (
from xarray.plot.accessor import DataArrayPlotAccessor
from xarray.plot.utils import _get_units_from_attrs
from xarray.util.deprecation_helpers import _deprecate_positional_args, deprecate_dims
def _to_dataset_whole(self, name: Hashable=None, shallow_copy: bool=True) -> Dataset:
    if name is None:
        name = self.name
    if name is None:
        raise ValueError('unable to convert unnamed DataArray to a Dataset without providing an explicit name')
    if name in self.coords:
        raise ValueError('cannot create a Dataset from a DataArray with the same name as one of its coordinates')
    variables = self._coords.copy()
    variables[name] = self.variable
    if shallow_copy:
        for k in variables:
            variables[k] = variables[k].copy(deep=False)
    indexes = self._indexes
    coord_names = set(self._coords)
    return Dataset._construct_direct(variables, coord_names, indexes=indexes)