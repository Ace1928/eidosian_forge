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
@pytest.mark.parametrize(('offset', 'invalid_month', 'exception'), [(YearBegin, 0, ValueError), (YearEnd, 0, ValueError), (YearBegin, 13, ValueError), (YearEnd, 13, ValueError), (YearBegin, 1.5, TypeError), (YearEnd, 1.5, TypeError), (QuarterBegin, 0, ValueError), (QuarterEnd, 0, ValueError), (QuarterBegin, 1.5, TypeError), (QuarterEnd, 1.5, TypeError), (QuarterBegin, 13, ValueError), (QuarterEnd, 13, ValueError)], ids=_id_func)
def test_year_offset_constructor_invalid_month(offset, invalid_month, exception):
    with pytest.raises(exception):
        offset(month=invalid_month)