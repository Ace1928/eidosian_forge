import asyncio
from functools import partial
from .base import SQLiteExecutionContext
from .pysqlite import SQLiteDialect_pysqlite
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class AsyncAdaptFallback_aiosqlite_connection(AsyncAdapt_aiosqlite_connection):
    __slots__ = ()
    await_ = staticmethod(await_fallback)