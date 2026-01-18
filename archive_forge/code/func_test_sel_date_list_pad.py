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
@pytest.mark.parametrize('sel_kwargs', [{'method': 'pad'}, {'method': 'pad', 'tolerance': timedelta(days=365)}])
def test_sel_date_list_pad(da, date_type, index, sel_kwargs):
    expected = xr.DataArray([2, 2], coords=[[index[1], index[1]]], dims=['time'])
    result = da.sel(time=[date_type(1, 3, 1), date_type(1, 4, 1)], **sel_kwargs)
    assert_identical(result, expected)