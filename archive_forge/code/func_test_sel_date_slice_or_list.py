from __future__ import annotations
import pickle
from datetime import timedelta
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import (
from xarray.tests import (
from xarray.tests.test_coding_times import (
@requires_cftime
def test_sel_date_slice_or_list(da, index, date_type):
    expected = xr.DataArray([1, 2], coords=[index[:2]], dims=['time'])
    result = da.sel(time=slice(date_type(1, 1, 1), date_type(1, 12, 30)))
    assert_identical(result, expected)
    result = da.sel(time=[date_type(1, 1, 1), date_type(1, 2, 1)])
    assert_identical(result, expected)