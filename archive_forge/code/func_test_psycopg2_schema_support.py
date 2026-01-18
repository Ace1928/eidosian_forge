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
@pytest.mark.db
def test_psycopg2_schema_support(postgresql_psycopg2_engine):
    conn = postgresql_psycopg2_engine
    df = DataFrame({'col1': [1, 2], 'col2': [0.1, 0.2], 'col3': ['a', 'n']})
    with conn.connect() as con:
        with con.begin():
            con.exec_driver_sql('DROP SCHEMA IF EXISTS other CASCADE;')
            con.exec_driver_sql('CREATE SCHEMA other;')
    assert df.to_sql(name='test_schema_public', con=conn, index=False) == 2
    assert df.to_sql(name='test_schema_public_explicit', con=conn, index=False, schema='public') == 2
    assert df.to_sql(name='test_schema_other', con=conn, index=False, schema='other') == 2
    res1 = sql.read_sql_table('test_schema_public', conn)
    tm.assert_frame_equal(df, res1)
    res2 = sql.read_sql_table('test_schema_public_explicit', conn)
    tm.assert_frame_equal(df, res2)
    res3 = sql.read_sql_table('test_schema_public_explicit', conn, schema='public')
    tm.assert_frame_equal(df, res3)
    res4 = sql.read_sql_table('test_schema_other', conn, schema='other')
    tm.assert_frame_equal(df, res4)
    msg = 'Table test_schema_other not found'
    with pytest.raises(ValueError, match=msg):
        sql.read_sql_table('test_schema_other', conn, schema='public')
    with conn.connect() as con:
        with con.begin():
            con.exec_driver_sql('DROP SCHEMA IF EXISTS other CASCADE;')
            con.exec_driver_sql('CREATE SCHEMA other;')
    assert df.to_sql(name='test_schema_other', con=conn, schema='other', index=False) == 2
    df.to_sql(name='test_schema_other', con=conn, schema='other', index=False, if_exists='replace')
    assert df.to_sql(name='test_schema_other', con=conn, schema='other', index=False, if_exists='append') == 2
    res = sql.read_sql_table('test_schema_other', conn, schema='other')
    tm.assert_frame_equal(concat([df, df], ignore_index=True), res)