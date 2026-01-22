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
class NetCDF4ArrayWrapper(BaseNetCDF4Array):
    __slots__ = ()

    def get_array(self, needs_lock=True):
        ds = self.datastore._acquire(needs_lock)
        variable = ds.variables[self.variable_name]
        variable.set_auto_maskandscale(False)
        with suppress(AttributeError):
            variable.set_auto_chartostring(False)
        return variable

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.OUTER, self._getitem)

    def _getitem(self, key):
        if self.datastore.is_remote:
            getitem = functools.partial(robust_getitem, catch=RuntimeError)
        else:
            getitem = operator.getitem
        try:
            with self.datastore.lock:
                original_array = self.get_array(needs_lock=False)
                array = getitem(original_array, key)
        except IndexError:
            msg = 'The indexing operation you are attempting to perform is not valid on netCDF4.Variable object. Try loading your data into memory first by calling .load().'
            raise IndexError(msg)
        return array