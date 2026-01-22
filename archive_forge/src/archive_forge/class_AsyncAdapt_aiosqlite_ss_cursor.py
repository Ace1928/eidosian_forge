import asyncio
from functools import partial
from .base import SQLiteExecutionContext
from .pysqlite import SQLiteDialect_pysqlite
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class AsyncAdapt_aiosqlite_ss_cursor(AsyncAdapt_aiosqlite_cursor):
    __slots__ = '_cursor'
    server_side = True

    def __init__(self, *arg, **kw):
        super().__init__(*arg, **kw)
        self._cursor = None

    def close(self):
        if self._cursor is not None:
            self.await_(self._cursor.close())
            self._cursor = None

    def fetchone(self):
        return self.await_(self._cursor.fetchone())

    def fetchmany(self, size=None):
        if size is None:
            size = self.arraysize
        return self.await_(self._cursor.fetchmany(size=size))

    def fetchall(self):
        return self.await_(self._cursor.fetchall())