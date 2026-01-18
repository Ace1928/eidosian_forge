from __future__ import annotations
from itertools import product
from typing import Callable, Literal
import numpy as np
import pandas as pd
import pytest
from xarray import CFTimeIndex
from xarray.coding.cftime_offsets import (
from xarray.coding.frequencies import infer_freq
from xarray.core.dataarray import DataArray
from xarray.tests import (
@pytest.mark.parametrize(('offset', 'expected_date_args'), [(Day(n=2), (1, 1, 1)), (Hour(n=2), (1, 1, 2, 22)), (Minute(n=2), (1, 1, 2, 23, 58)), (Second(n=2), (1, 1, 2, 23, 59, 58)), (Millisecond(n=2), (1, 1, 2, 23, 59, 59, 998000)), (Microsecond(n=2), (1, 1, 2, 23, 59, 59, 999998))], ids=_id_func)
def test_rsub_sub_monthly(offset, expected_date_args, calendar):
    date_type = get_date_type(calendar)
    initial = date_type(1, 1, 3)
    expected = date_type(*expected_date_args)
    result = initial - offset
    assert result == expected