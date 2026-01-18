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
def test_read_sql_string_inference(sqlite_engine):
    conn = sqlite_engine
    pytest.importorskip('pyarrow')
    table = 'test'
    df = DataFrame({'a': ['x', 'y']})
    df.to_sql(table, con=conn, index=False, if_exists='replace')
    with pd.option_context('future.infer_string', True):
        result = read_sql_table(table, conn)
    dtype = 'string[pyarrow_numpy]'
    expected = DataFrame({'a': ['x', 'y']}, dtype=dtype, columns=Index(['a'], dtype=dtype))
    tm.assert_frame_equal(result, expected)