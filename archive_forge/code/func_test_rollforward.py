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
@pytest.mark.parametrize(('offset', 'initial_date_args', 'partial_expected_date_args'), [(YearBegin(), (1, 3, 1), (2, 1)), (YearBegin(), (1, 1, 1), (1, 1)), (YearBegin(n=2), (1, 3, 1), (2, 1)), (YearBegin(n=2, month=2), (1, 3, 1), (2, 2)), (YearEnd(), (1, 3, 1), (1, 12)), (YearEnd(n=2), (1, 3, 1), (1, 12)), (YearEnd(n=2, month=2), (1, 3, 1), (2, 2)), (YearEnd(n=2, month=4), (1, 4, 30), (1, 4)), (QuarterBegin(), (1, 3, 2), (1, 6)), (QuarterBegin(), (1, 4, 1), (1, 6)), (QuarterBegin(n=2), (1, 4, 1), (1, 6)), (QuarterBegin(n=2, month=2), (1, 4, 1), (1, 5)), (QuarterEnd(), (1, 3, 1), (1, 3)), (QuarterEnd(n=2), (1, 3, 1), (1, 3)), (QuarterEnd(n=2, month=2), (1, 3, 1), (1, 5)), (QuarterEnd(n=2, month=4), (1, 4, 30), (1, 4)), (MonthBegin(), (1, 3, 2), (1, 4)), (MonthBegin(), (1, 3, 1), (1, 3)), (MonthBegin(n=2), (1, 3, 2), (1, 4)), (MonthEnd(), (1, 3, 2), (1, 3)), (MonthEnd(), (1, 4, 30), (1, 4)), (MonthEnd(n=2), (1, 3, 2), (1, 3)), (Day(), (1, 3, 2, 1), (1, 3, 2, 1)), (Hour(), (1, 3, 2, 1, 1), (1, 3, 2, 1, 1)), (Minute(), (1, 3, 2, 1, 1, 1), (1, 3, 2, 1, 1, 1)), (Second(), (1, 3, 2, 1, 1, 1, 1), (1, 3, 2, 1, 1, 1, 1)), (Millisecond(), (1, 3, 2, 1, 1, 1, 1000), (1, 3, 2, 1, 1, 1, 1000)), (Microsecond(), (1, 3, 2, 1, 1, 1, 1), (1, 3, 2, 1, 1, 1, 1))], ids=_id_func)
def test_rollforward(calendar, offset, initial_date_args, partial_expected_date_args):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    if isinstance(offset, (MonthBegin, QuarterBegin, YearBegin)):
        expected_date_args = partial_expected_date_args + (1,)
    elif isinstance(offset, (MonthEnd, QuarterEnd, YearEnd)):
        reference_args = partial_expected_date_args + (1,)
        reference = date_type(*reference_args)
        expected_date_args = partial_expected_date_args + (_days_in_month(reference),)
    else:
        expected_date_args = partial_expected_date_args
    expected = date_type(*expected_date_args)
    result = offset.rollforward(initial)
    assert result == expected