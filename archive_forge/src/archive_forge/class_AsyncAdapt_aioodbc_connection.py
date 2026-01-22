from __future__ import annotations
from typing import TYPE_CHECKING
from .asyncio import AsyncAdapt_dbapi_connection
from .asyncio import AsyncAdapt_dbapi_cursor
from .asyncio import AsyncAdapt_dbapi_ss_cursor
from .asyncio import AsyncAdaptFallback_dbapi_connection
from .pyodbc import PyODBCConnector
from .. import pool
from .. import util
from ..util.concurrency import await_fallback
from ..util.concurrency import await_only
class AsyncAdapt_aioodbc_connection(AsyncAdapt_dbapi_connection):
    _cursor_cls = AsyncAdapt_aioodbc_cursor
    _ss_cursor_cls = AsyncAdapt_aioodbc_ss_cursor
    __slots__ = ()

    @property
    def autocommit(self):
        return self._connection.autocommit

    @autocommit.setter
    def autocommit(self, value):
        self._connection._conn.autocommit = value

    def cursor(self, server_side=False):
        if self._connection.closed:
            raise self.dbapi.ProgrammingError('Attempt to use a closed connection.')
        return super().cursor(server_side=server_side)

    def rollback(self):
        if not self._connection.closed:
            super().rollback()

    def commit(self):
        if not self._connection.closed:
            super().commit()

    def close(self):
        if not self._connection.closed:
            super().close()