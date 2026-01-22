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
def _infer_coords_and_dims(shape: tuple[int, ...], coords: Sequence[Sequence | pd.Index | DataArray] | Mapping | None, dims: str | Iterable[Hashable] | None) -> tuple[Mapping[Hashable, Any], tuple[Hashable, ...]]:
    """All the logic for creating a new DataArray"""
    if coords is not None and (not utils.is_dict_like(coords)) and (len(coords) != len(shape)):
        raise ValueError(f'coords is not dict-like, but it has {len(coords)} items, which does not match the {len(shape)} dimensions of the data')
    if isinstance(dims, str):
        dims = (dims,)
    elif dims is None:
        dims = [f'dim_{n}' for n in range(len(shape))]
        if coords is not None and len(coords) == len(shape):
            if utils.is_dict_like(coords):
                dims = list(coords.keys())
            else:
                for n, (dim, coord) in enumerate(zip(dims, coords)):
                    coord = as_variable(coord, name=dims[n], auto_convert=False).to_index_variable()
                    dims[n] = coord.name
    dims_tuple = tuple(dims)
    if len(dims_tuple) != len(shape):
        raise ValueError(f'different number of dimensions on data and dims: {len(shape)} vs {len(dims_tuple)}')
    for d in dims_tuple:
        if not hashable(d):
            raise TypeError(f'Dimension {d} is not hashable')
    new_coords: Mapping[Hashable, Any]
    if isinstance(coords, Coordinates):
        new_coords = coords
    else:
        new_coords = {}
        if utils.is_dict_like(coords):
            for k, v in coords.items():
                new_coords[k] = as_variable(v, name=k, auto_convert=False)
                if new_coords[k].dims == (k,):
                    new_coords[k] = new_coords[k].to_index_variable()
        elif coords is not None:
            for dim, coord in zip(dims_tuple, coords):
                var = as_variable(coord, name=dim, auto_convert=False)
                var.dims = (dim,)
                new_coords[dim] = var.to_index_variable()
    _check_coords_dims(shape, new_coords, dims_tuple)
    return (new_coords, dims_tuple)