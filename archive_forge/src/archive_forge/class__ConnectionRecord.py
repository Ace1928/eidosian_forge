from __future__ import annotations
from collections import deque
import dataclasses
from enum import Enum
import threading
import time
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref
from .. import event
from .. import exc
from .. import log
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
class _ConnectionRecord(ConnectionPoolEntry):
    """Maintains a position in a connection pool which references a pooled
    connection.

    This is an internal object used by the :class:`_pool.Pool` implementation
    to provide context management to a DBAPI connection maintained by
    that :class:`_pool.Pool`.   The public facing interface for this class
    is described by the :class:`.ConnectionPoolEntry` class.  See that
    class for public API details.

    .. seealso::

        :class:`.ConnectionPoolEntry`

        :class:`.PoolProxiedConnection`

    """
    __slots__ = ('__pool', 'fairy_ref', 'finalize_callback', 'fresh', 'starttime', 'dbapi_connection', '__weakref__', '__dict__')
    finalize_callback: Deque[Callable[[DBAPIConnection], None]]
    fresh: bool
    fairy_ref: Optional[weakref.ref[_ConnectionFairy]]
    starttime: float

    def __init__(self, pool: Pool, connect: bool=True):
        self.fresh = False
        self.fairy_ref = None
        self.starttime = 0
        self.dbapi_connection = None
        self.__pool = pool
        if connect:
            self.__connect()
        self.finalize_callback = deque()
    dbapi_connection: Optional[DBAPIConnection]

    @property
    def driver_connection(self) -> Optional[Any]:
        if self.dbapi_connection is None:
            return None
        else:
            return self.__pool._dialect.get_driver_connection(self.dbapi_connection)

    @property
    @util.deprecated('2.0', "The _ConnectionRecord.connection attribute is deprecated; please use 'driver_connection'")
    def connection(self) -> Optional[DBAPIConnection]:
        return self.dbapi_connection
    _soft_invalidate_time: float = 0

    @util.ro_memoized_property
    def info(self) -> _InfoType:
        return {}

    @util.ro_memoized_property
    def record_info(self) -> Optional[_InfoType]:
        return {}

    @classmethod
    def checkout(cls, pool: Pool) -> _ConnectionFairy:
        if TYPE_CHECKING:
            rec = cast(_ConnectionRecord, pool._do_get())
        else:
            rec = pool._do_get()
        try:
            dbapi_connection = rec.get_connection()
        except BaseException as err:
            with util.safe_reraise():
                rec._checkin_failed(err, _fairy_was_created=False)
            raise
        echo = pool._should_log_debug()
        fairy = _ConnectionFairy(pool, dbapi_connection, rec, echo)
        rec.fairy_ref = ref = weakref.ref(fairy, lambda ref: _finalize_fairy(None, rec, pool, ref, echo, transaction_was_reset=False) if _finalize_fairy is not None else None)
        _strong_ref_connection_records[ref] = rec
        if echo:
            pool.logger.debug('Connection %r checked out from pool', dbapi_connection)
        return fairy

    def _checkin_failed(self, err: BaseException, _fairy_was_created: bool=True) -> None:
        self.invalidate(e=err)
        self.checkin(_fairy_was_created=_fairy_was_created)

    def checkin(self, _fairy_was_created: bool=True) -> None:
        if self.fairy_ref is None and _fairy_was_created:
            util.warn('Double checkin attempted on %s' % self)
            return
        self.fairy_ref = None
        connection = self.dbapi_connection
        pool = self.__pool
        while self.finalize_callback:
            finalizer = self.finalize_callback.pop()
            if connection is not None:
                finalizer(connection)
        if pool.dispatch.checkin:
            pool.dispatch.checkin(connection, self)
        pool._return_conn(self)

    @property
    def in_use(self) -> bool:
        return self.fairy_ref is not None

    @property
    def last_connect_time(self) -> float:
        return self.starttime

    def close(self) -> None:
        if self.dbapi_connection is not None:
            self.__close()

    def invalidate(self, e: Optional[BaseException]=None, soft: bool=False) -> None:
        if self.dbapi_connection is None:
            return
        if soft:
            self.__pool.dispatch.soft_invalidate(self.dbapi_connection, self, e)
        else:
            self.__pool.dispatch.invalidate(self.dbapi_connection, self, e)
        if e is not None:
            self.__pool.logger.info('%sInvalidate connection %r (reason: %s:%s)', 'Soft ' if soft else '', self.dbapi_connection, e.__class__.__name__, e)
        else:
            self.__pool.logger.info('%sInvalidate connection %r', 'Soft ' if soft else '', self.dbapi_connection)
        if soft:
            self._soft_invalidate_time = time.time()
        else:
            self.__close(terminate=True)
            self.dbapi_connection = None

    def get_connection(self) -> DBAPIConnection:
        recycle = False
        if self.dbapi_connection is None:
            self.info.clear()
            self.__connect()
        elif self.__pool._recycle > -1 and time.time() - self.starttime > self.__pool._recycle:
            self.__pool.logger.info('Connection %r exceeded timeout; recycling', self.dbapi_connection)
            recycle = True
        elif self.__pool._invalidate_time > self.starttime:
            self.__pool.logger.info('Connection %r invalidated due to pool invalidation; ' + 'recycling', self.dbapi_connection)
            recycle = True
        elif self._soft_invalidate_time > self.starttime:
            self.__pool.logger.info('Connection %r invalidated due to local soft invalidation; ' + 'recycling', self.dbapi_connection)
            recycle = True
        if recycle:
            self.__close(terminate=True)
            self.info.clear()
            self.__connect()
        assert self.dbapi_connection is not None
        return self.dbapi_connection

    def _is_hard_or_soft_invalidated(self) -> bool:
        return self.dbapi_connection is None or self.__pool._invalidate_time > self.starttime or self._soft_invalidate_time > self.starttime

    def __close(self, *, terminate: bool=False) -> None:
        self.finalize_callback.clear()
        if self.__pool.dispatch.close:
            self.__pool.dispatch.close(self.dbapi_connection, self)
        assert self.dbapi_connection is not None
        self.__pool._close_connection(self.dbapi_connection, terminate=terminate)
        self.dbapi_connection = None

    def __connect(self) -> None:
        pool = self.__pool
        self.dbapi_connection = None
        try:
            self.starttime = time.time()
            self.dbapi_connection = connection = pool._invoke_creator(self)
            pool.logger.debug('Created new connection %r', connection)
            self.fresh = True
        except BaseException as e:
            with util.safe_reraise():
                pool.logger.debug('Error on connect(): %s', e)
        else:
            if pool.dispatch.first_connect:
                pool.dispatch.first_connect.for_modify(pool.dispatch).exec_once_unless_exception(self.dbapi_connection, self)
            pool.dispatch.connect.for_modify(pool.dispatch)._exec_w_sync_on_first_run(self.dbapi_connection, self)