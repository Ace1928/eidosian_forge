from __future__ import annotations
import copy
import itertools
import math
import numbers
import warnings
from collections.abc import Hashable, Mapping, Sequence
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, NoReturn, cast
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import xarray as xr  # only for Dataset and DataArray
from xarray.core import common, dtypes, duck_array_ops, indexing, nputils, ops, utils
from xarray.core.arithmetic import VariableArithmetic
from xarray.core.common import AbstractArray
from xarray.core.indexing import (
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
from xarray.namedarray.core import NamedArray, _raise_if_any_duplicate_dimensions
from xarray.namedarray.pycompat import integer_types, is_0d_dask_array, to_duck_array
def _getitem_with_mask(self, key, fill_value=dtypes.NA):
    """Index this Variable with -1 remapped to fill_value."""
    if fill_value is dtypes.NA:
        fill_value = dtypes.get_fill_value(self.dtype)
    dims, indexer, new_order = self._broadcast_indexes(key)
    if self.size:
        if is_duck_dask_array(self._data):
            actual_indexer = indexing.posify_mask_indexer(indexer)
        else:
            actual_indexer = indexer
        indexable = as_indexable(self._data)
        data = indexing.apply_indexer(indexable, actual_indexer)
        mask = indexing.create_mask(indexer, self.shape, data)
        data = duck_array_ops.where(np.logical_not(mask), data, fill_value)
    else:
        mask = indexing.create_mask(indexer, self.shape)
        data = np.broadcast_to(fill_value, getattr(mask, 'shape', ()))
    if new_order:
        data = duck_array_ops.moveaxis(data, range(len(new_order)), new_order)
    return self._finalize_indexing_result(dims, data)