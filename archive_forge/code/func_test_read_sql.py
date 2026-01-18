from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
def test_read_sql(db):
    from sqlalchemy import sql
    s = sql.select(sql.column('number'), sql.column('name')).select_from(sql.table('test'))
    out = read_sql(s, db, npartitions=2, index_col='number')
    assert_eq(out, df[['name']])
    data = read_sql_table('test', db, npartitions=2, index_col='number').compute()
    assert (data.name == df.name).all()
    assert data.index.name == 'number'
    assert_eq(data, df)