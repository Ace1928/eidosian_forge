import asyncio
from functools import partial
from .base import SQLiteExecutionContext
from .pysqlite import SQLiteDialect_pysqlite
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class AsyncAdapt_aiosqlite_cursor:
    __slots__ = ('_adapt_connection', '_connection', 'description', 'await_', '_rows', 'arraysize', 'rowcount', 'lastrowid')
    server_side = False

    def __init__(self, adapt_connection):
        self._adapt_connection = adapt_connection
        self._connection = adapt_connection._connection
        self.await_ = adapt_connection.await_
        self.arraysize = 1
        self.rowcount = -1
        self.description = None
        self._rows = []

    def close(self):
        self._rows[:] = []

    def execute(self, operation, parameters=None):
        try:
            _cursor = self.await_(self._connection.cursor())
            if parameters is None:
                self.await_(_cursor.execute(operation))
            else:
                self.await_(_cursor.execute(operation, parameters))
            if _cursor.description:
                self.description = _cursor.description
                self.lastrowid = self.rowcount = -1
                if not self.server_side:
                    self._rows = self.await_(_cursor.fetchall())
            else:
                self.description = None
                self.lastrowid = _cursor.lastrowid
                self.rowcount = _cursor.rowcount
            if not self.server_side:
                self.await_(_cursor.close())
            else:
                self._cursor = _cursor
        except Exception as error:
            self._adapt_connection._handle_exception(error)

    def executemany(self, operation, seq_of_parameters):
        try:
            _cursor = self.await_(self._connection.cursor())
            self.await_(_cursor.executemany(operation, seq_of_parameters))
            self.description = None
            self.lastrowid = _cursor.lastrowid
            self.rowcount = _cursor.rowcount
            self.await_(_cursor.close())
        except Exception as error:
            self._adapt_connection._handle_exception(error)

    def setinputsizes(self, *inputsizes):
        pass

    def __iter__(self):
        while self._rows:
            yield self._rows.pop(0)

    def fetchone(self):
        if self._rows:
            return self._rows.pop(0)
        else:
            return None

    def fetchmany(self, size=None):
        if size is None:
            size = self.arraysize
        retval = self._rows[0:size]
        self._rows[:] = self._rows[size:]
        return retval

    def fetchall(self):
        retval = self._rows[:]
        self._rows[:] = []
        return retval