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
class _DummyGroup(Generic[T_Xarray]):
    """Class for keeping track of grouped dimensions without coordinates.

    Should not be user visible.
    """
    __slots__ = ('name', 'coords', 'size', 'dataarray')

    def __init__(self, obj: T_Xarray, name: Hashable, coords) -> None:
        self.name = name
        self.coords = coords
        self.size = obj.sizes[name]

    @property
    def dims(self) -> tuple[Hashable]:
        return (self.name,)

    @property
    def ndim(self) -> Literal[1]:
        return 1

    @property
    def values(self) -> range:
        return range(self.size)

    @property
    def data(self) -> range:
        return range(self.size)

    def __array__(self) -> np.ndarray:
        return np.arange(self.size)

    @property
    def shape(self) -> tuple[int]:
        return (self.size,)

    @property
    def attrs(self) -> dict:
        return {}

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = key[0]
        return self.values[key]

    def to_index(self) -> pd.Index:
        return pd.Index(np.arange(self.size))

    def copy(self, deep: bool=True, data: Any=None):
        raise NotImplementedError

    def to_dataarray(self) -> DataArray:
        from xarray.core.dataarray import DataArray
        return DataArray(data=self.data, dims=(self.name,), coords=self.coords, name=self.name)

    def to_array(self) -> DataArray:
        """Deprecated version of to_dataarray."""
        return self.to_dataarray()