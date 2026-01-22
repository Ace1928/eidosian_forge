from contextlib import asynccontextmanager
from .pymysql import MySQLDialect_pymysql
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class AsyncAdapt_asyncmy_dbapi:

    def __init__(self, asyncmy):
        self.asyncmy = asyncmy
        self.paramstyle = 'format'
        self._init_dbapi_attributes()

    def _init_dbapi_attributes(self):
        for name in ('Warning', 'Error', 'InterfaceError', 'DataError', 'DatabaseError', 'OperationalError', 'InterfaceError', 'IntegrityError', 'ProgrammingError', 'InternalError', 'NotSupportedError'):
            setattr(self, name, getattr(self.asyncmy.errors, name))
    STRING = util.symbol('STRING')
    NUMBER = util.symbol('NUMBER')
    BINARY = util.symbol('BINARY')
    DATETIME = util.symbol('DATETIME')
    TIMESTAMP = util.symbol('TIMESTAMP')
    Binary = staticmethod(_Binary)

    def connect(self, *arg, **kw):
        async_fallback = kw.pop('async_fallback', False)
        creator_fn = kw.pop('async_creator_fn', self.asyncmy.connect)
        if util.asbool(async_fallback):
            return AsyncAdaptFallback_asyncmy_connection(self, await_fallback(creator_fn(*arg, **kw)))
        else:
            return AsyncAdapt_asyncmy_connection(self, await_only(creator_fn(*arg, **kw)))