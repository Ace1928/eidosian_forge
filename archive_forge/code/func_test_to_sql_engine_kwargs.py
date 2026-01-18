from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
def test_to_sql_engine_kwargs(caplog):
    ddf = dd.from_pandas(df, 2)
    with tmp_db_uri() as uri:
        ddf.to_sql('test', uri, engine_kwargs={'echo': False})
        logs = '\n'.join((r.message for r in caplog.records))
        assert logs == ''
        assert_eq(df, read_sql_table('test', uri, 'number'))
    with tmp_db_uri() as uri:
        ddf.to_sql('test', uri, engine_kwargs={'echo': True})
        logs = '\n'.join((r.message for r in caplog.records))
        assert 'CREATE' in logs
        assert 'INSERT' in logs
        assert_eq(df, read_sql_table('test', uri, 'number'))