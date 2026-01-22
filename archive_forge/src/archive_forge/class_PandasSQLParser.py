import contextlib
import json
import os
import warnings
from io import BytesIO, IOBase, TextIOWrapper
from typing import Any, NamedTuple
import fsspec
import numpy as np
import pandas
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.concat import union_categoricals
from pandas.io.common import infer_compression
from pandas.util._decorators import doc
from modin.config import MinPartitionSize
from modin.core.io.file_dispatcher import OpenFile
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.db_conn import ModinDatabaseConnection
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import ModinAssumptionError
@doc(_doc_pandas_parser_class, data_type='SQL queries or tables')
class PandasSQLParser(PandasParser):

    @staticmethod
    @doc(_doc_parse_func, parameters="sql : str or SQLAlchemy Selectable (select or text object)\n    SQL query to be executed or a table name.\ncon : SQLAlchemy connectable, str, or sqlite3 connection\n    Connection object to database.\nindex_col : str or list of str\n    Column(s) to set as index(MultiIndex).\nread_sql_engine : str\n    Underlying engine ('pandas' or 'connectorx') used for fetching query result.")
    def parse(sql, con, index_col, read_sql_engine, **kwargs):
        enable_cx = False
        if read_sql_engine == 'Connectorx':
            try:
                import connectorx as cx
                enable_cx = True
            except ImportError:
                warnings.warn("Switch to 'pandas.read_sql' since 'connectorx' is not installed, please run 'pip install connectorx'.")
        num_splits = kwargs.pop('num_splits', None)
        if isinstance(con, ModinDatabaseConnection):
            con = con.get_string() if enable_cx else con.get_connection()
        if num_splits is None:
            if enable_cx:
                return cx.read_sql(con, sql, index_col=index_col)
            return pandas.read_sql(sql, con, index_col=index_col, **kwargs)
        if enable_cx:
            df = cx.read_sql(con, sql, index_col=index_col)
        else:
            df = pandas.read_sql(sql, con, index_col=index_col, **kwargs)
        if index_col is None:
            index = len(df)
        else:
            index = df.index
        return _split_result_for_readers(1, num_splits, df) + [index, df.dtypes]