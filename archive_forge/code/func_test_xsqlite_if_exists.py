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
def test_xsqlite_if_exists(sqlite_buildin):
    df_if_exists_1 = DataFrame({'col1': [1, 2], 'col2': ['A', 'B']})
    df_if_exists_2 = DataFrame({'col1': [3, 4, 5], 'col2': ['C', 'D', 'E']})
    table_name = 'table_if_exists'
    sql_select = f'SELECT * FROM {table_name}'
    msg = "'notvalidvalue' is not valid for if_exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists='notvalidvalue')
    drop_table(table_name, sqlite_buildin)
    sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists='fail')
    msg = "Table 'table_if_exists' already exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists='fail')
    sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists='replace', index=False)
    assert tquery(sql_select, con=sqlite_buildin) == [(1, 'A'), (2, 'B')]
    assert sql.to_sql(frame=df_if_exists_2, con=sqlite_buildin, name=table_name, if_exists='replace', index=False) == 3
    assert tquery(sql_select, con=sqlite_buildin) == [(3, 'C'), (4, 'D'), (5, 'E')]
    drop_table(table_name, sqlite_buildin)
    assert sql.to_sql(frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists='fail', index=False) == 2
    assert tquery(sql_select, con=sqlite_buildin) == [(1, 'A'), (2, 'B')]
    assert sql.to_sql(frame=df_if_exists_2, con=sqlite_buildin, name=table_name, if_exists='append', index=False) == 3
    assert tquery(sql_select, con=sqlite_buildin) == [(1, 'A'), (2, 'B'), (3, 'C'), (4, 'D'), (5, 'E')]
    drop_table(table_name, sqlite_buildin)