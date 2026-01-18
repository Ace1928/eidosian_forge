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
@pytest.mark.parametrize(('year_month_args', 'sub_day_args', 'offset'), [((1, 1), (), MonthEnd()), ((1, 1), (1,), MonthEnd()), ((1, 12), (), QuarterEnd()), ((1, 1), (), QuarterEnd(month=1)), ((1, 12), (), YearEnd()), ((1, 1), (), YearEnd(month=1))], ids=_id_func)
def test_onOffset_month_or_quarter_or_year_end(calendar, year_month_args, sub_day_args, offset):
    date_type = get_date_type(calendar)
    reference_args = year_month_args + (1,)
    reference = date_type(*reference_args)
    date_args = year_month_args + (_days_in_month(reference),) + sub_day_args
    date = date_type(*date_args)
    result = offset.onOffset(date)
    assert result