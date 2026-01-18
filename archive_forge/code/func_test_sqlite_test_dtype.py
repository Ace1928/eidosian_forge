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
def test_sqlite_test_dtype(sqlite_buildin):
    conn = sqlite_buildin
    cols = ['A', 'B']
    data = [(0.8, True), (0.9, None)]
    df = DataFrame(data, columns=cols)
    assert df.to_sql(name='dtype_test', con=conn) == 2
    assert df.to_sql(name='dtype_test2', con=conn, dtype={'B': 'STRING'}) == 2
    assert get_sqlite_column_type(conn, 'dtype_test', 'B') == 'INTEGER'
    assert get_sqlite_column_type(conn, 'dtype_test2', 'B') == 'STRING'
    msg = "B \\(<class 'bool'>\\) not a string"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name='error', con=conn, dtype={'B': bool})
    assert df.to_sql(name='single_dtype_test', con=conn, dtype='STRING') == 2
    assert get_sqlite_column_type(conn, 'single_dtype_test', 'A') == 'STRING'
    assert get_sqlite_column_type(conn, 'single_dtype_test', 'B') == 'STRING'