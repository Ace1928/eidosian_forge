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
@pytest.mark.parametrize(('string', 'date_args', 'reso'), [('1999', (1999, 1, 1), 'year'), ('199902', (1999, 2, 1), 'month'), ('19990202', (1999, 2, 2), 'day'), ('19990202T01', (1999, 2, 2, 1), 'hour'), ('19990202T0101', (1999, 2, 2, 1, 1), 'minute'), ('19990202T010156', (1999, 2, 2, 1, 1, 56), 'second')])
def test_parse_iso8601_with_reso(date_type, string, date_args, reso):
    expected_date = date_type(*date_args)
    expected_reso = reso
    result_date, result_reso = _parse_iso8601_with_reso(date_type, string)
    assert result_date == expected_date
    assert result_reso == expected_reso