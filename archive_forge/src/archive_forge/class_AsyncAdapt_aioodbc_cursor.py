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
class AsyncAdapt_aioodbc_cursor(AsyncAdapt_dbapi_cursor):
    __slots__ = ()

    def setinputsizes(self, *inputsizes):
        return self._cursor._impl.setinputsizes(*inputsizes)