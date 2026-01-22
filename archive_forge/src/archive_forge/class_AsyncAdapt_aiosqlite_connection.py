import asyncio
from functools import partial
from .base import SQLiteExecutionContext
from .pysqlite import SQLiteDialect_pysqlite
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class AsyncAdapt_aiosqlite_connection(AdaptedConnection):
    await_ = staticmethod(await_only)
    __slots__ = ('dbapi',)

    def __init__(self, dbapi, connection):
        self.dbapi = dbapi
        self._connection = connection

    @property
    def isolation_level(self):
        return self._connection.isolation_level

    @isolation_level.setter
    def isolation_level(self, value):

        def set_iso(connection, value):
            connection.isolation_level = value
        function = partial(set_iso, self._connection._conn, value)
        future = asyncio.get_event_loop().create_future()
        self._connection._tx.put_nowait((future, function))
        try:
            return self.await_(future)
        except Exception as error:
            self._handle_exception(error)

    def create_function(self, *args, **kw):
        try:
            self.await_(self._connection.create_function(*args, **kw))
        except Exception as error:
            self._handle_exception(error)

    def cursor(self, server_side=False):
        if server_side:
            return AsyncAdapt_aiosqlite_ss_cursor(self)
        else:
            return AsyncAdapt_aiosqlite_cursor(self)

    def execute(self, *args, **kw):
        return self.await_(self._connection.execute(*args, **kw))

    def rollback(self):
        try:
            self.await_(self._connection.rollback())
        except Exception as error:
            self._handle_exception(error)

    def commit(self):
        try:
            self.await_(self._connection.commit())
        except Exception as error:
            self._handle_exception(error)

    def close(self):
        try:
            self.await_(self._connection.close())
        except ValueError:
            pass
        except Exception as error:
            self._handle_exception(error)

    def _handle_exception(self, error):
        if isinstance(error, ValueError) and error.args[0] == 'no active connection':
            raise self.dbapi.sqlite.OperationalError('no active connection') from error
        else:
            raise error