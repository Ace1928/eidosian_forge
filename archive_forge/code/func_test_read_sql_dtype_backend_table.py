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
@pytest.mark.parametrize('conn', all_connectable)
@pytest.mark.parametrize('func', ['read_sql', 'read_sql_table'])
def test_read_sql_dtype_backend_table(conn, request, string_storage, func, dtype_backend, dtype_backend_data, dtype_backend_expected):
    if 'sqlite' in conn and 'adbc' not in conn:
        request.applymarker(pytest.mark.xfail(reason='SQLite actually returns proper boolean values via read_sql_table, but before pytest refactor was skipped'))
    conn_name = conn
    conn = request.getfixturevalue(conn)
    table = 'test'
    df = dtype_backend_data
    df.to_sql(name=table, con=conn, index=False, if_exists='replace')
    with pd.option_context('mode.string_storage', string_storage):
        result = getattr(pd, func)(table, conn, dtype_backend=dtype_backend)
    expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
    tm.assert_frame_equal(result, expected)
    if 'adbc' in conn_name:
        return
    with pd.option_context('mode.string_storage', string_storage):
        iterator = getattr(pd, func)(table, conn, dtype_backend=dtype_backend, chunksize=3)
        expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
        for result in iterator:
            tm.assert_frame_equal(result, expected)