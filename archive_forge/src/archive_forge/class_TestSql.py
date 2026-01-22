import contextlib
import csv
import inspect
import os
import sys
import unittest.mock as mock
from collections import defaultdict
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict
import fastparquet
import numpy as np
import pandas
import pandas._libs.lib as lib
import pyarrow as pa
import pyarrow.dataset
import pytest
import sqlalchemy as sa
from packaging import version
from pandas._testing import ensure_clean
from pandas.errors import ParserWarning
from scipy import sparse
from modin.config import (
from modin.db_conn import ModinDatabaseConnection, UnsupportedDatabaseException
from modin.pandas.io import from_arrow, from_dask, from_ray, to_pandas
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import (
from .utils import test_data as utils_test_data
from .utils import time_parsing_csv_path
from modin.config import NPartitions
@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestSql:

    @pytest.mark.parametrize('read_sql_engine', ['Pandas', 'Connectorx'])
    def test_read_sql(self, tmp_path, make_sql_connection, read_sql_engine):
        filename = get_unique_filename('.db')
        table = 'test_read_sql'
        conn = make_sql_connection(tmp_path / filename, table)
        query = f'select * from {table}'
        eval_io(fn_name='read_sql', sql=query, con=conn)
        eval_io(fn_name='read_sql', sql=query, con=conn, index_col='index')
        with warns_that_defaulting_to_pandas():
            pd.read_sql_query(query, conn)
        with warns_that_defaulting_to_pandas():
            pd.read_sql_table(table, conn)
        sqlalchemy_engine = sa.create_engine(conn)
        eval_io(fn_name='read_sql', sql=query, con=sqlalchemy_engine)
        sqlalchemy_connection = sqlalchemy_engine.connect()
        eval_io(fn_name='read_sql', sql=query, con=sqlalchemy_connection)
        old_sql_engine = ReadSqlEngine.get()
        ReadSqlEngine.put(read_sql_engine)
        if ReadSqlEngine.get() == 'Connectorx':
            modin_df = pd.read_sql(sql=query, con=conn)
        else:
            modin_df = pd.read_sql(sql=query, con=ModinDatabaseConnection('sqlalchemy', conn))
        ReadSqlEngine.put(old_sql_engine)
        pandas_df = pandas.read_sql(sql=query, con=sqlalchemy_connection)
        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize('dtype_backend', [lib.no_default, 'numpy_nullable', 'pyarrow'])
    def test_read_sql_dtype_backend(self, tmp_path, make_sql_connection, dtype_backend):
        filename = get_unique_filename(extension='db')
        table = 'test_read_sql_dtype_backend'
        conn = make_sql_connection(tmp_path / filename, table)
        query = f'select * from {table}'

        def comparator(df1, df2):
            df_equals(df1, df2)
            df_equals(df1.dtypes, df2.dtypes)
        eval_io(fn_name='read_sql', sql=query, con=conn, dtype_backend=dtype_backend, comparator=comparator)

    @pytest.mark.skipif(not TestReadFromSqlServer.get(), reason='Skip the test when the test SQL server is not set up.')
    def test_read_sql_from_sql_server(self):
        table_name = 'test_1000x256'
        query = f'SELECT * FROM {table_name}'
        sqlalchemy_connection_string = 'mssql+pymssql://sa:Strong.Pwd-123@0.0.0.0:1433/master'
        pandas_df_to_read = pandas.DataFrame(np.arange(1000 * 256).reshape(1000, 256)).add_prefix('col')
        pandas_df_to_read.to_sql(table_name, sqlalchemy_connection_string, if_exists='replace')
        modin_df = pd.read_sql(query, ModinDatabaseConnection('sqlalchemy', sqlalchemy_connection_string))
        pandas_df = pandas.read_sql(query, sqlalchemy_connection_string)
        df_equals(modin_df, pandas_df)

    @pytest.mark.skipif(not TestReadFromPostgres.get(), reason='Skip the test when the postgres server is not set up.')
    def test_read_sql_from_postgres(self):
        table_name = 'test_1000x256'
        query = f'SELECT * FROM {table_name}'
        connection = 'postgresql://sa:Strong.Pwd-123@localhost:2345/postgres'
        pandas_df_to_read = pandas.DataFrame(np.arange(1000 * 256).reshape(1000, 256)).add_prefix('col')
        pandas_df_to_read.to_sql(table_name, connection, if_exists='replace')
        modin_df = pd.read_sql(query, ModinDatabaseConnection('psycopg2', connection))
        pandas_df = pandas.read_sql(query, connection)
        df_equals(modin_df, pandas_df)

    def test_invalid_modin_database_connections(self):
        with pytest.raises(UnsupportedDatabaseException):
            ModinDatabaseConnection('unsupported_database')

    def test_read_sql_with_chunksize(self, make_sql_connection):
        filename = get_unique_filename(extension='db')
        table = 'test_read_sql_with_chunksize'
        conn = make_sql_connection(filename, table)
        query = f'select * from {table}'
        pandas_gen = pandas.read_sql(query, conn, chunksize=10)
        modin_gen = pd.read_sql(query, conn, chunksize=10)
        for modin_df, pandas_df in zip(modin_gen, pandas_gen):
            df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize('index', [False, True])
    @pytest.mark.parametrize('conn_type', ['str', 'sqlalchemy', 'sqlalchemy+connect'])
    def test_to_sql(self, tmp_path, make_sql_connection, index, conn_type):
        table_name = f'test_to_sql_{str(index)}'
        modin_df, pandas_df = create_test_dfs(TEST_DATA)
        conn = make_sql_connection(tmp_path / f'{table_name}_modin.db')
        if conn_type.startswith('sqlalchemy'):
            conn = sa.create_engine(conn)
            if conn_type == 'sqlalchemy+connect':
                conn = conn.connect()
        modin_df.to_sql(table_name, conn, index=index)
        df_modin_sql = pandas.read_sql(table_name, con=conn, index_col='index' if index else None)
        conn = make_sql_connection(tmp_path / f'{table_name}_pandas.db')
        if conn_type.startswith('sqlalchemy'):
            conn = sa.create_engine(conn)
            if conn_type == 'sqlalchemy+connect':
                conn = conn.connect()
        pandas_df.to_sql(table_name, conn, index=index)
        df_pandas_sql = pandas.read_sql(table_name, con=conn, index_col='index' if index else None)
        assert df_modin_sql.sort_index().equals(df_pandas_sql.sort_index())