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
def thin(self, indexers: Mapping[Any, int] | int | None=None, **indexers_kwargs: Any) -> Self:
    """Returns a new dataset with each array indexed along every `n`-th
        value for the specified dimension(s)

        Parameters
        ----------
        indexers : dict or int
            A dict with keys matching dimensions and integer values `n`
            or a single integer `n` applied over all dimensions.
            One of indexers or indexers_kwargs must be provided.
        **indexers_kwargs : {dim: n, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Examples
        --------
        >>> x_arr = np.arange(0, 26)
        >>> x_arr
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
               17, 18, 19, 20, 21, 22, 23, 24, 25])
        >>> x = xr.DataArray(
        ...     np.reshape(x_arr, (2, 13)),
        ...     dims=("x", "y"),
        ...     coords={"x": [0, 1], "y": np.arange(0, 13)},
        ... )
        >>> x_ds = xr.Dataset({"foo": x})
        >>> x_ds
        <xarray.Dataset> Size: 328B
        Dimensions:  (x: 2, y: 13)
        Coordinates:
          * x        (x) int64 16B 0 1
          * y        (y) int64 104B 0 1 2 3 4 5 6 7 8 9 10 11 12
        Data variables:
            foo      (x, y) int64 208B 0 1 2 3 4 5 6 7 8 ... 17 18 19 20 21 22 23 24 25

        >>> x_ds.thin(3)
        <xarray.Dataset> Size: 88B
        Dimensions:  (x: 1, y: 5)
        Coordinates:
          * x        (x) int64 8B 0
          * y        (y) int64 40B 0 3 6 9 12
        Data variables:
            foo      (x, y) int64 40B 0 3 6 9 12
        >>> x.thin({"x": 2, "y": 5})
        <xarray.DataArray (x: 1, y: 3)> Size: 24B
        array([[ 0,  5, 10]])
        Coordinates:
          * x        (x) int64 8B 0
          * y        (y) int64 24B 0 5 10

        See Also
        --------
        Dataset.head
        Dataset.tail
        DataArray.thin
        """
    if not indexers_kwargs and (not isinstance(indexers, int)) and (not is_dict_like(indexers)):
        raise TypeError('indexers must be either dict-like or a single integer')
    if isinstance(indexers, int):
        indexers = {dim: indexers for dim in self.dims}
    indexers = either_dict_or_kwargs(indexers, indexers_kwargs, 'thin')
    for k, v in indexers.items():
        if not isinstance(v, int):
            raise TypeError(f'expected integer type indexer for dimension {k!r}, found {type(v)!r}')
        elif v < 0:
            raise ValueError(f'expected positive integer as indexer for dimension {k!r}, found {v}')
        elif v == 0:
            raise ValueError('step cannot be zero')
    indexers_slices = {k: slice(None, None, val) for k, val in indexers.items()}
    return self.isel(indexers_slices)