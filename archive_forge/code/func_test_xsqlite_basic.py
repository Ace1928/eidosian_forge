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
def test_xsqlite_basic(sqlite_buildin):
    frame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    assert sql.to_sql(frame, name='test_table', con=sqlite_buildin, index=False) == 10
    result = sql.read_sql('select * from test_table', sqlite_buildin)
    result.index = frame.index
    expected = frame
    tm.assert_frame_equal(result, frame)
    frame['txt'] = ['a'] * len(frame)
    frame2 = frame.copy()
    new_idx = Index(np.arange(len(frame2)), dtype=np.int64) + 10
    frame2['Idx'] = new_idx.copy()
    assert sql.to_sql(frame2, name='test_table2', con=sqlite_buildin, index=False) == 10
    result = sql.read_sql('select * from test_table2', sqlite_buildin, index_col='Idx')
    expected = frame.copy()
    expected.index = new_idx
    expected.index.name = 'Idx'
    tm.assert_frame_equal(expected, result)