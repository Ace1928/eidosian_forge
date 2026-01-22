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
def _broadcast_indexes_outer(self, key):
    dims = tuple((k.dims[0] if isinstance(k, Variable) else dim for k, dim in zip(key, self.dims) if not isinstance(k, integer_types) and (not is_0d_dask_array(k))))
    new_key = []
    for k in key:
        if isinstance(k, Variable):
            k = k.data
        if not isinstance(k, BASIC_INDEXING_TYPES):
            if not is_duck_array(k):
                k = np.asarray(k)
            if k.size == 0:
                k = k.astype(int)
            elif k.dtype.kind == 'b':
                k, = np.nonzero(k)
        new_key.append(k)
    return (dims, OuterIndexer(tuple(new_key)), None)