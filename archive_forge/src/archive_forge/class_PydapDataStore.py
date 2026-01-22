from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray.backends.common import (
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.namedarray.pycompat import integer_types
class PydapDataStore(AbstractDataStore):
    """Store for accessing OpenDAP datasets with pydap.

    This store provides an alternative way to access OpenDAP datasets that may
    be useful if the netCDF4 library is not available.
    """

    def __init__(self, ds):
        """
        Parameters
        ----------
        ds : pydap DatasetType
        """
        self.ds = ds

    @classmethod
    def open(cls, url, application=None, session=None, output_grid=None, timeout=None, verify=None, user_charset=None):
        import pydap.client
        import pydap.lib
        if timeout is None:
            from pydap.lib import DEFAULT_TIMEOUT
            timeout = DEFAULT_TIMEOUT
        kwargs = {'url': url, 'application': application, 'session': session, 'output_grid': output_grid or True, 'timeout': timeout}
        if verify is not None:
            kwargs.update({'verify': verify})
        if user_charset is not None:
            kwargs.update({'user_charset': user_charset})
        ds = pydap.client.open_url(**kwargs)
        return cls(ds)

    def open_store_variable(self, var):
        data = indexing.LazilyIndexedArray(PydapArrayWrapper(var))
        return Variable(var.dimensions, data, _fix_attributes(var.attributes))

    def get_variables(self):
        return FrozenDict(((k, self.open_store_variable(self.ds[k])) for k in self.ds.keys()))

    def get_attrs(self):
        return Frozen(_fix_attributes(self.ds.attributes))

    def get_dimensions(self):
        return Frozen(self.ds.dimensions)