from __future__ import annotations
import itertools
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, Union
import pandas as pd
from xarray.core import dtypes
from xarray.core.concat import concat
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.merge import merge
from xarray.core.utils import iterate_nested
def _infer_concat_order_from_coords(datasets):
    concat_dims = []
    tile_ids = [() for ds in datasets]
    ds0 = datasets[0]
    for dim in ds0.dims:
        if dim in ds0:
            indexes = [ds._indexes.get(dim) for ds in datasets]
            if any((index is None for index in indexes)):
                raise ValueError('Every dimension needs a coordinate for inferring concatenation order')
            indexes = [index.to_pandas_index() for index in indexes]
            if not all((index.equals(indexes[0]) for index in indexes[1:])):
                concat_dims.append(dim)
                if all((index.is_monotonic_increasing for index in indexes)):
                    ascending = True
                elif all((index.is_monotonic_decreasing for index in indexes)):
                    ascending = False
                else:
                    raise ValueError(f'Coordinate variable {dim} is neither monotonically increasing nor monotonically decreasing on all datasets')
                if any((index.size == 0 for index in indexes)):
                    raise ValueError('Cannot handle size zero dimensions')
                first_items = pd.Index([index[0] for index in indexes])
                series = first_items.to_series()
                _ensure_same_types(series, dim)
                rank = series.rank(method='dense', ascending=ascending, numeric_only=False)
                order = rank.astype(int).values - 1
                tile_ids = [tile_id + (position,) for tile_id, position in zip(tile_ids, order)]
    if len(datasets) > 1 and (not concat_dims):
        raise ValueError('Could not find any dimension coordinates to use to order the datasets for concatenation')
    combined_ids = dict(zip(tile_ids, datasets))
    return (combined_ids, concat_dims)