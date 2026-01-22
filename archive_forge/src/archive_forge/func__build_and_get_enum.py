from __future__ import annotations
import functools
import operator
import os
from collections.abc import Iterable
from contextlib import suppress
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray import coding
from xarray.backends.common import (
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import (
from xarray.backends.netcdf3 import encode_nc3_attr_value, encode_nc3_variable
from xarray.backends.store import StoreBackendEntrypoint
from xarray.coding.variables import pop_to
from xarray.core import indexing
from xarray.core.utils import (
from xarray.core.variable import Variable
def _build_and_get_enum(self, var_name: str, dtype: np.dtype, enum_name: str, enum_dict: dict[str, int]) -> Any:
    """
        Add or get the netCDF4 Enum based on the dtype in encoding.
        The return type should be ``netCDF4.EnumType``,
        but we avoid importing netCDF4 globally for performances.
        """
    if enum_name not in self.ds.enumtypes:
        return self.ds.createEnumType(dtype, enum_name, enum_dict)
    datatype = self.ds.enumtypes[enum_name]
    if datatype.enum_dict != enum_dict:
        error_msg = f"Cannot save variable `{var_name}` because an enum `{enum_name}` already exists in the Dataset but have a different definition. To fix this error, make sure each variable have a uniquely named enum in their `encoding['dtype'].metadata` or, if they should share the same enum type, make sure the enums are identical."
        raise ValueError(error_msg)
    return datatype