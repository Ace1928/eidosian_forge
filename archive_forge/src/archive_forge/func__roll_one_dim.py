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
def _roll_one_dim(self, dim, count):
    axis = self.get_axis_num(dim)
    count %= self.shape[axis]
    if count != 0:
        indices = [slice(-count, None), slice(None, -count)]
    else:
        indices = [slice(None)]
    arrays = [self[(slice(None),) * axis + (idx,)].data for idx in indices]
    data = duck_array_ops.concatenate(arrays, axis)
    if is_duck_dask_array(data):
        data = data.rechunk(self.data.chunks)
    return self._replace(data=data)