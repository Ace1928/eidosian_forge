from __future__ import annotations
from numbers import Number
import numpy as np
import pytest
import xarray as xr
from xarray.backends.api import _get_default_engine
from xarray.tests import (
class CustomBackend(xr.backends.BackendEntrypoint):

    def open_dataset(self, filename_or_obj, drop_variables=None, **kwargs) -> xr.Dataset:
        return expected.copy(deep=True)