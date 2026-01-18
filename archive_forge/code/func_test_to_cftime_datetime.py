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
@pytest.mark.parametrize(('argument', 'expected_date_args'), [('2000-01-01', (2000, 1, 1)), ((2000, 1, 1), (2000, 1, 1))], ids=_id_func)
def test_to_cftime_datetime(calendar, argument, expected_date_args):
    date_type = get_date_type(calendar)
    expected = date_type(*expected_date_args)
    if isinstance(argument, tuple):
        argument = date_type(*argument)
    result = to_cftime_datetime(argument, calendar=calendar)
    assert result == expected