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
@pytest.mark.parametrize('calendar', ['all_leap', '360_day'])
def test_to_datetimeindex_feb_29(calendar):
    index = xr.cftime_range('2001-02-28', periods=2, calendar=calendar)
    with pytest.raises(ValueError, match='29'):
        index.to_datetimeindex()