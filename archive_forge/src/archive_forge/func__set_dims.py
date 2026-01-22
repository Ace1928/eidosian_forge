from __future__ import annotations
import functools
import operator
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Final, Generic, TypeVar, cast, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes
from xarray.core.indexes import (
from xarray.core.types import T_Alignable
from xarray.core.utils import is_dict_like, is_full_slice
from xarray.core.variable import Variable, as_compatible_data, calculate_dimensions
def _set_dims(var):
    var_dims_map = dims_map.copy()
    for dim in exclude:
        with suppress(ValueError):
            var_dims_map[dim] = var.shape[var.dims.index(dim)]
    return var.set_dims(var_dims_map)