from __future__ import annotations
import warnings
import numpy as np
import pytest
import xarray as xr
from xarray.tests import has_dask
class WarningVariable(xr.Variable):

    @property
    def dims(self):
        warnings.warn('warning in test')
        return super().dims

    def __array__(self):
        warnings.warn('warning in test')
        return super().__array__()