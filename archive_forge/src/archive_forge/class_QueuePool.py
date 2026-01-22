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
class QueuePool(Pool):
    """A :class:`_pool.Pool`
    that imposes a limit on the number of open connections.

    :class:`.QueuePool` is the default pooling implementation used for
    all :class:`_engine.Engine` objects other than SQLite with a ``:memory:``
    database.

    The :class:`.QueuePool` class **is not compatible** with asyncio and
    :func:`_asyncio.create_async_engine`.  The
    :class:`.AsyncAdaptedQueuePool` class is used automatically when
    using :func:`_asyncio.create_async_engine`, if no other kind of pool
    is specified.

    .. seealso::

        :class:`.AsyncAdaptedQueuePool`

    """
    _is_asyncio = False
    _queue_class: Type[sqla_queue.QueueCommon[ConnectionPoolEntry]] = sqla_queue.Queue
    _pool: sqla_queue.QueueCommon[ConnectionPoolEntry]

    def __init__(self, creator: Union[_CreatorFnType, _CreatorWRecFnType], pool_size: int=5, max_overflow: int=10, timeout: float=30.0, use_lifo: bool=False, **kw: Any):
        """
        Construct a QueuePool.

        :param creator: a callable function that returns a DB-API
          connection object, same as that of :paramref:`_pool.Pool.creator`.

        :param pool_size: The size of the pool to be maintained,
          defaults to 5. This is the largest number of connections that
          will be kept persistently in the pool. Note that the pool
          begins with no connections; once this number of connections
          is requested, that number of connections will remain.
          ``pool_size`` can be set to 0 to indicate no size limit; to
          disable pooling, use a :class:`~sqlalchemy.pool.NullPool`
          instead.

        :param max_overflow: The maximum overflow size of the
          pool. When the number of checked-out connections reaches the
          size set in pool_size, additional connections will be
          returned up to this limit. When those additional connections
          are returned to the pool, they are disconnected and
          discarded. It follows then that the total number of
          simultaneous connections the pool will allow is pool_size +
          `max_overflow`, and the total number of "sleeping"
          connections the pool will allow is pool_size. `max_overflow`
          can be set to -1 to indicate no overflow limit; no limit
          will be placed on the total number of concurrent
          connections. Defaults to 10.

        :param timeout: The number of seconds to wait before giving up
          on returning a connection. Defaults to 30.0. This can be a float
          but is subject to the limitations of Python time functions which
          may not be reliable in the tens of milliseconds.

        :param use_lifo: use LIFO (last-in-first-out) when retrieving
          connections instead of FIFO (first-in-first-out). Using LIFO, a
          server-side timeout scheme can reduce the number of connections used
          during non-peak periods of use.   When planning for server-side
          timeouts, ensure that a recycle or pre-ping strategy is in use to
          gracefully handle stale connections.

          .. versionadded:: 1.3

          .. seealso::

            :ref:`pool_use_lifo`

            :ref:`pool_disconnects`

        :param \\**kw: Other keyword arguments including
          :paramref:`_pool.Pool.recycle`, :paramref:`_pool.Pool.echo`,
          :paramref:`_pool.Pool.reset_on_return` and others are passed to the
          :class:`_pool.Pool` constructor.

        """
        Pool.__init__(self, creator, **kw)
        self._pool = self._queue_class(pool_size, use_lifo=use_lifo)
        self._overflow = 0 - pool_size
        self._max_overflow = -1 if pool_size == 0 else max_overflow
        self._timeout = timeout
        self._overflow_lock = threading.Lock()

    def _do_return_conn(self, record: ConnectionPoolEntry) -> None:
        try:
            self._pool.put(record, False)
        except sqla_queue.Full:
            try:
                record.close()
            finally:
                self._dec_overflow()

    def _do_get(self) -> ConnectionPoolEntry:
        use_overflow = self._max_overflow > -1
        wait = use_overflow and self._overflow >= self._max_overflow
        try:
            return self._pool.get(wait, self._timeout)
        except sqla_queue.Empty:
            pass
        if use_overflow and self._overflow >= self._max_overflow:
            if not wait:
                return self._do_get()
            else:
                raise exc.TimeoutError('QueuePool limit of size %d overflow %d reached, connection timed out, timeout %0.2f' % (self.size(), self.overflow(), self._timeout), code='3o7r')
        if self._inc_overflow():
            try:
                return self._create_connection()
            except:
                with util.safe_reraise():
                    self._dec_overflow()
                raise
        else:
            return self._do_get()

    def _inc_overflow(self) -> bool:
        if self._max_overflow == -1:
            self._overflow += 1
            return True
        with self._overflow_lock:
            if self._overflow < self._max_overflow:
                self._overflow += 1
                return True
            else:
                return False

    def _dec_overflow(self) -> Literal[True]:
        if self._max_overflow == -1:
            self._overflow -= 1
            return True
        with self._overflow_lock:
            self._overflow -= 1
            return True

    def recreate(self) -> QueuePool:
        self.logger.info('Pool recreating')
        return self.__class__(self._creator, pool_size=self._pool.maxsize, max_overflow=self._max_overflow, pre_ping=self._pre_ping, use_lifo=self._pool.use_lifo, timeout=self._timeout, recycle=self._recycle, echo=self.echo, logging_name=self._orig_logging_name, reset_on_return=self._reset_on_return, _dispatch=self.dispatch, dialect=self._dialect)

    def dispose(self) -> None:
        while True:
            try:
                conn = self._pool.get(False)
                conn.close()
            except sqla_queue.Empty:
                break
        self._overflow = 0 - self.size()
        self.logger.info('Pool disposed. %s', self.status())

    def status(self) -> str:
        return 'Pool size: %d  Connections in pool: %d Current Overflow: %d Current Checked out connections: %d' % (self.size(), self.checkedin(), self.overflow(), self.checkedout())

    def size(self) -> int:
        return self._pool.maxsize

    def timeout(self) -> float:
        return self._timeout

    def checkedin(self) -> int:
        return self._pool.qsize()

    def overflow(self) -> int:
        return self._overflow if self._pool.maxsize else 0

    def checkedout(self) -> int:
        return self._pool.maxsize - self._pool.qsize() + self._overflow