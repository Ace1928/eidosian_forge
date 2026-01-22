from __future__ import annotations
from abc import (
from contextlib import (
from datetime import (
from functools import partial
import re
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import get_option
from pandas.core.api import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime
class SQLiteTable(SQLTable):
    """
    Patch the SQLTable for fallback support.
    Instead of a table variable just use the Create Table statement.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_date_adapters()

    def _register_date_adapters(self) -> None:
        import sqlite3

        def _adapt_time(t) -> str:
            return f'{t.hour:02d}:{t.minute:02d}:{t.second:02d}.{t.microsecond:06d}'
        adapt_date_iso = lambda val: val.isoformat()
        adapt_datetime_iso = lambda val: val.isoformat(' ')
        sqlite3.register_adapter(time, _adapt_time)
        sqlite3.register_adapter(date, adapt_date_iso)
        sqlite3.register_adapter(datetime, adapt_datetime_iso)
        convert_date = lambda val: date.fromisoformat(val.decode())
        convert_timestamp = lambda val: datetime.fromisoformat(val.decode())
        sqlite3.register_converter('date', convert_date)
        sqlite3.register_converter('timestamp', convert_timestamp)

    def sql_schema(self) -> str:
        return str(';\n'.join(self.table))

    def _execute_create(self) -> None:
        with self.pd_sql.run_transaction() as conn:
            for stmt in self.table:
                conn.execute(stmt)

    def insert_statement(self, *, num_rows: int) -> str:
        names = list(map(str, self.frame.columns))
        wld = '?'
        escape = _get_valid_sqlite_name
        if self.index is not None:
            for idx in self.index[::-1]:
                names.insert(0, idx)
        bracketed_names = [escape(column) for column in names]
        col_names = ','.join(bracketed_names)
        row_wildcards = ','.join([wld] * len(names))
        wildcards = ','.join([f'({row_wildcards})' for _ in range(num_rows)])
        insert_statement = f'INSERT INTO {escape(self.name)} ({col_names}) VALUES {wildcards}'
        return insert_statement

    def _execute_insert(self, conn, keys, data_iter) -> int:
        data_list = list(data_iter)
        conn.executemany(self.insert_statement(num_rows=1), data_list)
        return conn.rowcount

    def _execute_insert_multi(self, conn, keys, data_iter) -> int:
        data_list = list(data_iter)
        flattened_data = [x for row in data_list for x in row]
        conn.execute(self.insert_statement(num_rows=len(data_list)), flattened_data)
        return conn.rowcount

    def _create_table_setup(self):
        """
        Return a list of SQL statements that creates a table reflecting the
        structure of a DataFrame.  The first entry will be a CREATE TABLE
        statement while the rest will be CREATE INDEX statements.
        """
        column_names_and_types = self._get_column_names_and_types(self._sql_type_name)
        escape = _get_valid_sqlite_name
        create_tbl_stmts = [escape(cname) + ' ' + ctype for cname, ctype, _ in column_names_and_types]
        if self.keys is not None and len(self.keys):
            if not is_list_like(self.keys):
                keys = [self.keys]
            else:
                keys = self.keys
            cnames_br = ', '.join([escape(c) for c in keys])
            create_tbl_stmts.append(f'CONSTRAINT {self.name}_pk PRIMARY KEY ({cnames_br})')
        if self.schema:
            schema_name = self.schema + '.'
        else:
            schema_name = ''
        create_stmts = ['CREATE TABLE ' + schema_name + escape(self.name) + ' (\n' + ',\n  '.join(create_tbl_stmts) + '\n)']
        ix_cols = [cname for cname, _, is_index in column_names_and_types if is_index]
        if len(ix_cols):
            cnames = '_'.join(ix_cols)
            cnames_br = ','.join([escape(c) for c in ix_cols])
            create_stmts.append('CREATE INDEX ' + escape('ix_' + self.name + '_' + cnames) + 'ON ' + escape(self.name) + ' (' + cnames_br + ')')
        return create_stmts

    def _sql_type_name(self, col):
        dtype: DtypeArg = self.dtype or {}
        if is_dict_like(dtype):
            dtype = cast(dict, dtype)
            if col.name in dtype:
                return dtype[col.name]
        col_type = lib.infer_dtype(col, skipna=True)
        if col_type == 'timedelta64':
            warnings.warn("the 'timedelta' type is not supported, and will be written as integer values (ns frequency) to the database.", UserWarning, stacklevel=find_stack_level())
            col_type = 'integer'
        elif col_type == 'datetime64':
            col_type = 'datetime'
        elif col_type == 'empty':
            col_type = 'string'
        elif col_type == 'complex':
            raise ValueError('Complex datatypes not supported')
        if col_type not in _SQL_TYPES:
            col_type = 'string'
        return _SQL_TYPES[col_type]