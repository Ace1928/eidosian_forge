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
def test_sqlite_datetime_date(sqlite_buildin):
    conn = sqlite_buildin
    df = DataFrame([date(2014, 1, 1), date(2014, 1, 2)], columns=['a'])
    assert df.to_sql(name='test_date', con=conn, index=False) == 2
    res = read_sql_query('SELECT * FROM test_date', conn)
    tm.assert_frame_equal(res, df.astype(str))