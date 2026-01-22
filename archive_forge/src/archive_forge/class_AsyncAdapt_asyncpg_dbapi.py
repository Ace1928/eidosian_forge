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
class AsyncAdapt_asyncpg_dbapi:

    def __init__(self, asyncpg):
        self.asyncpg = asyncpg
        self.paramstyle = 'numeric_dollar'

    def connect(self, *arg, **kw):
        async_fallback = kw.pop('async_fallback', False)
        creator_fn = kw.pop('async_creator_fn', self.asyncpg.connect)
        prepared_statement_cache_size = kw.pop('prepared_statement_cache_size', 100)
        prepared_statement_name_func = kw.pop('prepared_statement_name_func', None)
        if util.asbool(async_fallback):
            return AsyncAdaptFallback_asyncpg_connection(self, await_fallback(creator_fn(*arg, **kw)), prepared_statement_cache_size=prepared_statement_cache_size, prepared_statement_name_func=prepared_statement_name_func)
        else:
            return AsyncAdapt_asyncpg_connection(self, await_only(creator_fn(*arg, **kw)), prepared_statement_cache_size=prepared_statement_cache_size, prepared_statement_name_func=prepared_statement_name_func)

    class Error(Exception):
        pass

    class Warning(Exception):
        pass

    class InterfaceError(Error):
        pass

    class DatabaseError(Error):
        pass

    class InternalError(DatabaseError):
        pass

    class OperationalError(DatabaseError):
        pass

    class ProgrammingError(DatabaseError):
        pass

    class IntegrityError(DatabaseError):
        pass

    class DataError(DatabaseError):
        pass

    class NotSupportedError(DatabaseError):
        pass

    class InternalServerError(InternalError):
        pass

    class InvalidCachedStatementError(NotSupportedError):

        def __init__(self, message):
            super().__init__(message + ' (SQLAlchemy asyncpg dialect will now invalidate all prepared caches in response to this exception)')
    STRING = util.symbol('STRING')
    NUMBER = util.symbol('NUMBER')
    DATETIME = util.symbol('DATETIME')

    @util.memoized_property
    def _asyncpg_error_translate(self):
        import asyncpg
        return {asyncpg.exceptions.IntegrityConstraintViolationError: self.IntegrityError, asyncpg.exceptions.PostgresError: self.Error, asyncpg.exceptions.SyntaxOrAccessError: self.ProgrammingError, asyncpg.exceptions.InterfaceError: self.InterfaceError, asyncpg.exceptions.InvalidCachedStatementError: self.InvalidCachedStatementError, asyncpg.exceptions.InternalServerError: self.InternalServerError}

    def Binary(self, value):
        return value