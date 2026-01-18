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
def test_sqlite_notna_dtype(sqlite_buildin):
    conn = sqlite_buildin
    cols = {'Bool': Series([True, None]), 'Date': Series([datetime(2012, 5, 1), None]), 'Int': Series([1, None], dtype='object'), 'Float': Series([1.1, None])}
    df = DataFrame(cols)
    tbl = 'notna_dtype_test'
    assert df.to_sql(name=tbl, con=conn) == 2
    assert get_sqlite_column_type(conn, tbl, 'Bool') == 'INTEGER'
    assert get_sqlite_column_type(conn, tbl, 'Date') == 'TIMESTAMP'
    assert get_sqlite_column_type(conn, tbl, 'Int') == 'INTEGER'
    assert get_sqlite_column_type(conn, tbl, 'Float') == 'REAL'