from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
@pytest.mark.parametrize('op, expected', [('shift', {'time': [None, None, Timestamp('2019-01-01 12:00:00'), Timestamp('2019-01-01 12:30:00'), None, None]}), ('bfill', {'time': [Timestamp('2019-01-01 12:00:00'), Timestamp('2019-01-01 12:30:00'), Timestamp('2019-01-01 14:00:00'), Timestamp('2019-01-01 14:30:00'), Timestamp('2019-01-01 14:00:00'), Timestamp('2019-01-01 14:30:00')]}), ('ffill', {'time': [Timestamp('2019-01-01 12:00:00'), Timestamp('2019-01-01 12:30:00'), Timestamp('2019-01-01 12:00:00'), Timestamp('2019-01-01 12:30:00'), Timestamp('2019-01-01 14:00:00'), Timestamp('2019-01-01 14:30:00')]})])
def test_shift_bfill_ffill_tz(tz_naive_fixture, op, expected):
    tz = tz_naive_fixture
    data = {'id': ['A', 'B', 'A', 'B', 'A', 'B'], 'time': [Timestamp('2019-01-01 12:00:00'), Timestamp('2019-01-01 12:30:00'), None, None, Timestamp('2019-01-01 14:00:00'), Timestamp('2019-01-01 14:30:00')]}
    df = DataFrame(data).assign(time=lambda x: x.time.dt.tz_localize(tz))
    grouped = df.groupby('id')
    result = getattr(grouped, op)()
    expected = DataFrame(expected).assign(time=lambda x: x.time.dt.tz_localize(tz))
    tm.assert_frame_equal(result, expected)