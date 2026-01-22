from the connection pool, such as when using an ORM :class:`.Session` where
from working correctly.  The pysqlite DBAPI driver has several
import math
import os
import re
from .base import DATE
from .base import DATETIME
from .base import SQLiteDialect
from ... import exc
from ... import pool
from ... import types as sqltypes
from ... import util
class _SQLiteDialect_pysqlite_numeric(SQLiteDialect_pysqlite):
    """numeric dialect for testing only

    internal use only.  This dialect is **NOT** supported by SQLAlchemy
    and may change at any time.

    """
    supports_statement_cache = True
    default_paramstyle = 'numeric'
    driver = 'pysqlite_numeric'
    _first_bind = ':1'
    _not_in_statement_regexp = None

    def __init__(self, *arg, **kw):
        kw.setdefault('paramstyle', 'numeric')
        super().__init__(*arg, **kw)

    def create_connect_args(self, url):
        arg, opts = super().create_connect_args(url)
        opts['factory'] = self._fix_sqlite_issue_99953()
        return (arg, opts)

    def _fix_sqlite_issue_99953(self):
        import sqlite3
        first_bind = self._first_bind
        if self._not_in_statement_regexp:
            nis = self._not_in_statement_regexp

            def _test_sql(sql):
                m = nis.search(sql)
                assert not m, f'Found {nis.pattern!r} in {sql!r}'
        else:

            def _test_sql(sql):
                pass

        def _numeric_param_as_dict(parameters):
            if parameters:
                assert isinstance(parameters, tuple)
                return {str(idx): value for idx, value in enumerate(parameters, 1)}
            else:
                return ()

        class SQLiteFix99953Cursor(sqlite3.Cursor):

            def execute(self, sql, parameters=()):
                _test_sql(sql)
                if first_bind in sql:
                    parameters = _numeric_param_as_dict(parameters)
                return super().execute(sql, parameters)

            def executemany(self, sql, parameters):
                _test_sql(sql)
                if first_bind in sql:
                    parameters = [_numeric_param_as_dict(p) for p in parameters]
                return super().executemany(sql, parameters)

        class SQLiteFix99953Connection(sqlite3.Connection):

            def cursor(self, factory=None):
                if factory is None:
                    factory = SQLiteFix99953Cursor
                return super().cursor(factory=factory)

            def execute(self, sql, parameters=()):
                _test_sql(sql)
                if first_bind in sql:
                    parameters = _numeric_param_as_dict(parameters)
                return super().execute(sql, parameters)

            def executemany(self, sql, parameters):
                _test_sql(sql)
                if first_bind in sql:
                    parameters = [_numeric_param_as_dict(p) for p in parameters]
                return super().executemany(sql, parameters)
        return SQLiteFix99953Connection