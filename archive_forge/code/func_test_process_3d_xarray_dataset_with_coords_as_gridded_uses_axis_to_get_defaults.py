import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_process_3d_xarray_dataset_with_coords_as_gridded_uses_axis_to_get_defaults(self):
    import xarray as xr
    kwargs = self.default_kwargs
    kwargs.update(gridded=True)
    data, x, y, by, groupby = process_xarray(data=self.ds, **kwargs)
    assert isinstance(data, xr.Dataset)
    assert list(data.data_vars.keys()) == ['air']
    assert x == 'lon'
    assert y == 'lat'
    assert not by
    assert groupby == ['time']