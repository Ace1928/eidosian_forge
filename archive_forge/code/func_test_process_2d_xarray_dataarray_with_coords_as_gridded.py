import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_process_2d_xarray_dataarray_with_coords_as_gridded(self):
    import xarray as xr
    da = xr.DataArray(data=np.random.randn(4, 5), coords={'y': [3, 4, 5, 6, 7]}, dims=['x', 'y'])
    kwargs = self.default_kwargs
    kwargs.update(gridded=True)
    data, x, y, by, groupby = process_xarray(data=da, **kwargs)
    assert isinstance(data, xr.Dataset)
    assert list(data.data_vars.keys()) == ['value']
    assert x == 'y'
    assert y == 'x'
    assert not by
    assert not groupby