import operator
import sys
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType
import numpy as np
import pandas as pd
import param
from ..core.data import PandasInterface
from ..core.dimension import Dimension
from ..core.util import flatten, resolve_dependent_value, unique_iterator
class xr_dim(dim):
    """
    A subclass of dim which provides access to the xarray DataArray
    namespace along with tab-completion and type coercion allowing
    the expression to be applied on any gridded dataset.
    """
    namespace = 'xarray'
    _accessor = 'xr'

    def __init__(self, obj, *args, **kwargs):
        try:
            import xarray as xr
        except ImportError:
            raise ImportError('XArray could not be imported, dim().xr requires the xarray to be available.') from None
        super().__init__(obj, *args, **kwargs)
        self._ns = xr.DataArray

    def interface_applies(self, dataset, coerce):
        return dataset.interface.gridded and (coerce or dataset.interface.datatype == 'xarray')

    def _compute_data(self, data, drop_index, compute):
        if drop_index:
            data = data.data
        if hasattr(data, 'compute') and compute:
            data = data.compute()
        return data

    def _coerce(self, dataset):
        if self.interface_applies(dataset, coerce=False):
            return dataset
        return dataset.clone(datatype=['xarray'])