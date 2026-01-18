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
@pytest.mark.parametrize('calendar', _ALL_CALENDARS)
def test_to_datetimeindex_out_of_range(calendar):
    index = xr.cftime_range('0001', periods=5, calendar=calendar)
    with pytest.raises(ValueError, match='0001'):
        index.to_datetimeindex()