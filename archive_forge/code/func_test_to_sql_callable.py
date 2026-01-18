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
@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_to_sql_callable(conn, test_frame1, request):
    conn = request.getfixturevalue(conn)
    check = []

    def sample(pd_table, conn, keys, data_iter):
        check.append(1)
        data = [dict(zip(keys, row)) for row in data_iter]
        conn.execute(pd_table.table.insert(), data)
    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        pandasSQL.to_sql(test_frame1, 'test_frame', method=sample)
        assert pandasSQL.has_table('test_frame')
    assert check == [1]
    assert count_rows(conn, 'test_frame') == len(test_frame1)