from __future__ import annotations
import threading
import traceback
import typing
from typing import Any
from typing import cast
from typing import List
from typing import Optional
from typing import Set
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import weakref
from .base import _AsyncConnDialect
from .base import _ConnectionFairy
from .base import _ConnectionRecord
from .base import _CreatorFnType
from .base import _CreatorWRecFnType
from .base import ConnectionPoolEntry
from .base import Pool
from .base import PoolProxiedConnection
from .. import exc
from .. import util
from ..util import chop_traceback
from ..util import queue as sqla_queue
from ..util.typing import Literal
class StaticPool(Pool):
    """A Pool of exactly one connection, used for all requests.

    Reconnect-related functions such as ``recycle`` and connection
    invalidation (which is also used to support auto-reconnect) are only
    partially supported right now and may not yield good results.

    The :class:`.StaticPool` class **is compatible** with asyncio and
    :func:`_asyncio.create_async_engine`.

    """

    @util.memoized_property
    def connection(self) -> _ConnectionRecord:
        return _ConnectionRecord(self)

    def status(self) -> str:
        return 'StaticPool'

    def dispose(self) -> None:
        if 'connection' in self.__dict__ and self.connection.dbapi_connection is not None:
            self.connection.close()
            del self.__dict__['connection']

    def recreate(self) -> StaticPool:
        self.logger.info('Pool recreating')
        return self.__class__(creator=self._creator, recycle=self._recycle, reset_on_return=self._reset_on_return, pre_ping=self._pre_ping, echo=self.echo, logging_name=self._orig_logging_name, _dispatch=self.dispatch, dialect=self._dialect)

    def _transfer_from(self, other_static_pool: StaticPool) -> None:

        def creator(rec: ConnectionPoolEntry) -> DBAPIConnection:
            conn = other_static_pool.connection.dbapi_connection
            assert conn is not None
            return conn
        self._invoke_creator = creator

    def _create_connection(self) -> ConnectionPoolEntry:
        raise NotImplementedError()

    def _do_return_conn(self, record: ConnectionPoolEntry) -> None:
        pass

    def _do_get(self) -> ConnectionPoolEntry:
        rec = self.connection
        if rec._is_hard_or_soft_invalidated():
            del self.__dict__['connection']
            rec = self.connection
        return rec