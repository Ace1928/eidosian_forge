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
def test_to_sql_save_index(conn, request):
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason='ADBC implementation does not create index', strict=True))
    conn_name = conn
    conn = request.getfixturevalue(conn)
    df = DataFrame.from_records([(1, 2.1, 'line1'), (2, 1.5, 'line2')], columns=['A', 'B', 'C'], index=['A'])
    tbl_name = 'test_to_sql_saves_index'
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(df, tbl_name) == 2
    if conn_name in {'sqlite_buildin', 'sqlite_str'}:
        ixs = sql.read_sql_query(f"SELECT * FROM sqlite_master WHERE type = 'index' AND tbl_name = '{tbl_name}'", conn)
        ix_cols = []
        for ix_name in ixs.name:
            ix_info = sql.read_sql_query(f'PRAGMA index_info({ix_name})', conn)
            ix_cols.append(ix_info.name.tolist())
    else:
        from sqlalchemy import inspect
        insp = inspect(conn)
        ixs = insp.get_indexes(tbl_name)
        ix_cols = [i['column_names'] for i in ixs]
    assert ix_cols == [['A']]