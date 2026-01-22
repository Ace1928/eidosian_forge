from __future__ import annotations
import pickle
import pytest
import xarray as xr
from xarray.tests import assert_identical
@xr.register_dataset_accessor('demo')
@xr.register_dataarray_accessor('demo')
class DemoAccessor:
    """Demo accessor."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def foo(self):
        return 'bar'