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
def test_read_table_columns(conn, request, test_frame1):
    conn_name = conn
    if conn_name == 'sqlite_buildin':
        request.applymarker(pytest.mark.xfail(reason='Not Implemented'))
    conn = request.getfixturevalue(conn)
    sql.to_sql(test_frame1, 'test_frame', conn)
    cols = ['A', 'B']
    result = sql.read_sql_table('test_frame', conn, columns=cols)
    assert result.columns.tolist() == cols