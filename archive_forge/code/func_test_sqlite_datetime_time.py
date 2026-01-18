from __future__ import annotations
import contextlib
from contextlib import closing
import csv
from datetime import (
from io import StringIO
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING
import uuid
import numpy as np
import pytest
from pandas._libs import lib
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.util.version import Version
from pandas.io import sql
from pandas.io.sql import (
@pytest.mark.parametrize('tz_aware', [False, True])
def test_sqlite_datetime_time(tz_aware, sqlite_buildin):
    conn = sqlite_buildin
    if not tz_aware:
        tz_times = [time(9, 0, 0), time(9, 1, 30)]
    else:
        tz_dt = date_range('2013-01-01 09:00:00', periods=2, tz='US/Pacific')
        tz_times = Series(tz_dt.to_pydatetime()).map(lambda dt: dt.timetz())
    df = DataFrame(tz_times, columns=['a'])
    assert df.to_sql(name='test_time', con=conn, index=False) == 2
    res = read_sql_query('SELECT * FROM test_time', conn)
    expected = df.map(lambda _: _.strftime('%H:%M:%S.%f'))
    tm.assert_frame_equal(res, expected)