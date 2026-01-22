from __future__ import annotations
import copy
import datetime
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.cftime_offsets import _new_to_legacy_freq
from xarray.core import dtypes, duck_array_ops, nputils, ops
from xarray.core._aggregations import (
from xarray.core.alignment import align
from xarray.core.arithmetic import DataArrayGroupbyArithmetic, DatasetGroupbyArithmetic
from xarray.core.common import ImplementsArrayReduce, ImplementsDatasetReduce
from xarray.core.concat import concat
from xarray.core.formatting import format_array_flat
from xarray.core.indexes import (
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import (
from xarray.core.utils import (
from xarray.core.variable import IndexVariable, Variable
from xarray.util.deprecation_helpers import _deprecate_positional_args
def _resolve_group(obj: T_DataWithCoords, group: T_Group | Hashable) -> T_Group:
    from xarray.core.dataarray import DataArray
    error_msg = "the group variable's length does not match the length of this variable along its dimensions"
    newgroup: T_Group
    if isinstance(group, DataArray):
        try:
            align(obj, group, join='exact', copy=False)
        except ValueError:
            raise ValueError(error_msg)
        newgroup = group.copy(deep=False)
        newgroup.name = group.name or 'group'
    elif isinstance(group, IndexVariable):
        if group.ndim != 1:
            raise ValueError('Grouping by multi-dimensional IndexVariables is not allowed.Convert to and pass a DataArray instead.')
        group_dim, = group.dims
        if len(group) != obj.sizes[group_dim]:
            raise ValueError(error_msg)
        newgroup = DataArray(group)
    else:
        if not hashable(group):
            raise TypeError(f'`group` must be an xarray.DataArray or the name of an xarray variable or dimension. Received {group!r} instead.')
        group_da: DataArray = obj[group]
        if group_da.name not in obj._indexes and group_da.name in obj.dims:
            newgroup = _DummyGroup(obj, group_da.name, group_da.coords)
        else:
            newgroup = group_da
    if newgroup.size == 0:
        raise ValueError(f'{newgroup.name} must not be empty')
    return newgroup