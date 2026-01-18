from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@requires_dask
@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('ds', (2,), indirect=True)
@pytest.mark.parametrize('name', ('mean', 'max'))
def test_raise_no_warning_dask_rolling_assert_close(self, ds, name) -> None:
    """
        This is a puzzle — I can't easily find the source of the warning. It
        requires `assert_allclose` to be run, for the `ds` param to be 2, and is
        different for `mean` and `max`. `sum` raises no warning.
        """
    ds = ds.chunk({'x': 4})
    rolling_obj = ds.rolling(time=4, x=3)
    actual = getattr(rolling_obj, name)()
    expected = getattr(getattr(ds.rolling(time=4), name)().rolling(x=3), name)()
    assert_allclose(actual, expected)