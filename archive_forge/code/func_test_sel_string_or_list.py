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
@pytest.mark.parametrize('sel_arg', list(SEL_STRING_OR_LIST_TESTS.values()), ids=list(SEL_STRING_OR_LIST_TESTS.keys()))
def test_sel_string_or_list(da, index, sel_arg):
    expected = xr.DataArray([1, 2], coords=[index[:2]], dims=['time'])
    result = da.sel(time=sel_arg)
    assert_identical(result, expected)