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
def _nested_combine(datasets, concat_dims, compat, data_vars, coords, ids, fill_value=dtypes.NA, join: JoinOptions='outer', combine_attrs: CombineAttrsOptions='drop'):
    if len(datasets) == 0:
        return Dataset()
    if not ids:
        combined_ids = _infer_concat_order_from_positions(datasets)
    else:
        combined_ids = dict(zip(ids, datasets))
    _check_shape_tile_ids(combined_ids)
    combined = _combine_nd(combined_ids, concat_dims, compat=compat, data_vars=data_vars, coords=coords, fill_value=fill_value, join=join, combine_attrs=combine_attrs)
    return combined