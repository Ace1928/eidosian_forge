from __future__ import annotations
import datetime as dt
import warnings
from collections.abc import Hashable, Sequence
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, get_args
import numpy as np
import pandas as pd
from xarray.core import utils
from xarray.core.common import _contains_datetime_like_objects, ones_like
from xarray.core.computation import apply_ufunc
from xarray.core.duck_array_ops import (
from xarray.core.options import _get_keep_attrs
from xarray.core.types import Interp1dOptions, InterpOptions
from xarray.core.utils import OrderedSet, is_scalar
from xarray.core.variable import Variable, broadcast_variables
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def _get_nan_block_lengths(obj: Dataset | DataArray | Variable, dim: Hashable, index: Variable):
    """
    Return an object where each NaN element in 'obj' is replaced by the
    length of the gap the element is in.
    """
    index = Variable([dim], index)
    arange = ones_like(obj) * index
    valid = obj.notnull()
    valid_arange = arange.where(valid)
    cumulative_nans = valid_arange.ffill(dim=dim).fillna(index[0])
    nan_block_lengths = cumulative_nans.diff(dim=dim, label='upper').reindex({dim: obj[dim]}).where(valid).bfill(dim=dim).where(~valid, 0).fillna(index[-1] - valid_arange.max(dim=[dim]))
    return nan_block_lengths