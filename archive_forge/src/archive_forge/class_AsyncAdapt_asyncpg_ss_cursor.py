from __future__ import annotations
import collections
import decimal
import json as _py_json
import re
import time
from . import json
from . import ranges
from .array import ARRAY as PGARRAY
from .base import _DECIMAL_TYPES
from .base import _FLOAT_TYPES
from .base import _INT_TYPES
from .base import ENUM
from .base import INTERVAL
from .base import OID
from .base import PGCompiler
from .base import PGDialect
from .base import PGExecutionContext
from .base import PGIdentifierPreparer
from .base import REGCLASS
from .base import REGCONFIG
from .types import BIT
from .types import BYTEA
from .types import CITEXT
from ... import exc
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...engine import processors
from ...sql import sqltypes
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class AsyncAdapt_asyncpg_ss_cursor(AsyncAdapt_asyncpg_cursor):
    server_side = True
    __slots__ = ('_rowbuffer',)

    def __init__(self, adapt_connection):
        super().__init__(adapt_connection)
        self._rowbuffer = None

    def close(self):
        self._cursor = None
        self._rowbuffer = None

    def _buffer_rows(self):
        new_rows = self._adapt_connection.await_(self._cursor.fetch(50))
        self._rowbuffer = collections.deque(new_rows)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._rowbuffer:
            self._buffer_rows()
        while True:
            while self._rowbuffer:
                yield self._rowbuffer.popleft()
            self._buffer_rows()
            if not self._rowbuffer:
                break

    def fetchone(self):
        if not self._rowbuffer:
            self._buffer_rows()
            if not self._rowbuffer:
                return None
        return self._rowbuffer.popleft()

    def fetchmany(self, size=None):
        if size is None:
            return self.fetchall()
        if not self._rowbuffer:
            self._buffer_rows()
        buf = list(self._rowbuffer)
        lb = len(buf)
        if size > lb:
            buf.extend(self._adapt_connection.await_(self._cursor.fetch(size - lb)))
        result = buf[0:size]
        self._rowbuffer = collections.deque(buf[size:])
        return result

    def fetchall(self):
        ret = list(self._rowbuffer) + list(self._adapt_connection.await_(self._all()))
        self._rowbuffer.clear()
        return ret

    async def _all(self):
        rows = []
        while True:
            batch = await self._cursor.fetch(1000)
            if batch:
                rows.extend(batch)
                continue
            else:
                break
        return rows

    def executemany(self, operation, seq_of_parameters):
        raise NotImplementedError("server side cursor doesn't support executemany yet")