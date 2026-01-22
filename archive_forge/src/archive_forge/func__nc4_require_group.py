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
def _nc4_require_group(ds, group, mode, create_group=_netcdf4_create_group):
    if group in {None, '', '/'}:
        return ds
    else:
        if not isinstance(group, str):
            raise ValueError('group must be a string or None')
        path = group.strip('/').split('/')
        for key in path:
            try:
                ds = ds.groups[key]
            except KeyError as e:
                if mode != 'r':
                    ds = create_group(ds, key)
                else:
                    raise OSError(f'group not found: {key}', e)
        return ds