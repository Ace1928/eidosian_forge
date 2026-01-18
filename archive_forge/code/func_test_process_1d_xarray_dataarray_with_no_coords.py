import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_process_1d_xarray_dataarray_with_no_coords(self):
    import xarray as xr
    import pandas as pd
    da = xr.DataArray(data=[1, 2, 3])
    data, x, y, by, groupby = process_xarray(data=da, **self.default_kwargs)
    assert isinstance(data, pd.DataFrame)
    assert x == 'index'
    assert y == ['value']
    assert not by
    assert not groupby