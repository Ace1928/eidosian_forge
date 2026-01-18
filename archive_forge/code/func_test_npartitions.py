from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
def test_npartitions(db):
    data = read_sql_table('test', db, columns=list(df.columns), npartitions=2, index_col='number')
    assert len(data.divisions) == 3
    assert (data.name.compute() == df.name).all()
    data = read_sql_table('test', db, columns=['name'], npartitions=6, index_col='number')
    assert_eq(data, df[['name']])
    data = read_sql_table('test', db, columns=list(df.columns), bytes_per_chunk='2 GiB', index_col='number')
    assert data.npartitions == 1
    assert (data.name.compute() == df.name).all()
    data_1 = read_sql_table('test', db, columns=list(df.columns), bytes_per_chunk=2 ** 30, index_col='number', head_rows=1)
    assert data_1.npartitions == 1
    assert (data_1.name.compute() == df.name).all()
    data = read_sql_table('test', db, columns=list(df.columns), bytes_per_chunk=250, index_col='number', head_rows=1)
    assert (data.memory_usage_per_partition(deep=True, index=True) < 400).compute().all()
    assert (data.name.compute() == df.name).all()