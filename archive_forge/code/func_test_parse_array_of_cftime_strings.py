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
def test_parse_array_of_cftime_strings():
    from cftime import DatetimeNoLeap
    strings = np.array([['2000-01-01', '2000-01-02'], ['2000-01-03', '2000-01-04']])
    expected = np.array([[DatetimeNoLeap(2000, 1, 1), DatetimeNoLeap(2000, 1, 2)], [DatetimeNoLeap(2000, 1, 3), DatetimeNoLeap(2000, 1, 4)]])
    result = _parse_array_of_cftime_strings(strings, DatetimeNoLeap)
    np.testing.assert_array_equal(result, expected)
    strings = np.array('2000-01-01')
    expected = np.array(DatetimeNoLeap(2000, 1, 1))
    result = _parse_array_of_cftime_strings(strings, DatetimeNoLeap)
    np.testing.assert_array_equal(result, expected)