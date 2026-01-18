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
@pytest.mark.parametrize(('reso', 'ex_start_args', 'ex_end_args'), [('day', (2, 2, 10), (2, 2, 10, 23, 59, 59, 999999)), ('hour', (2, 2, 10, 6), (2, 2, 10, 6, 59, 59, 999999)), ('minute', (2, 2, 10, 6, 2), (2, 2, 10, 6, 2, 59, 999999)), ('second', (2, 2, 10, 6, 2, 8), (2, 2, 10, 6, 2, 8, 999999))])
def test_parsed_string_to_bounds_sub_monthly(date_type, reso, ex_start_args, ex_end_args):
    parsed = date_type(2, 2, 10, 6, 2, 8, 123456)
    expected_start = date_type(*ex_start_args)
    expected_end = date_type(*ex_end_args)
    result_start, result_end = _parsed_string_to_bounds(date_type, reso, parsed)
    assert result_start == expected_start
    assert result_end == expected_end