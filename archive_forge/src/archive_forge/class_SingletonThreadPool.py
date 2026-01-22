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
class SingletonThreadPool(Pool):
    """A Pool that maintains one connection per thread.

    Maintains one connection per each thread, never moving a connection to a
    thread other than the one which it was created in.

    .. warning::  the :class:`.SingletonThreadPool` will call ``.close()``
       on arbitrary connections that exist beyond the size setting of
       ``pool_size``, e.g. if more unique **thread identities**
       than what ``pool_size`` states are used.   This cleanup is
       non-deterministic and not sensitive to whether or not the connections
       linked to those thread identities are currently in use.

       :class:`.SingletonThreadPool` may be improved in a future release,
       however in its current status it is generally used only for test
       scenarios using a SQLite ``:memory:`` database and is not recommended
       for production use.

    The :class:`.SingletonThreadPool` class **is not compatible** with asyncio
    and :func:`_asyncio.create_async_engine`.


    Options are the same as those of :class:`_pool.Pool`, as well as:

    :param pool_size: The number of threads in which to maintain connections
        at once.  Defaults to five.

    :class:`.SingletonThreadPool` is used by the SQLite dialect
    automatically when a memory-based database is used.
    See :ref:`sqlite_toplevel`.

    """
    _is_asyncio = False

    def __init__(self, creator: Union[_CreatorFnType, _CreatorWRecFnType], pool_size: int=5, **kw: Any):
        Pool.__init__(self, creator, **kw)
        self._conn = threading.local()
        self._fairy = threading.local()
        self._all_conns: Set[ConnectionPoolEntry] = set()
        self.size = pool_size

    def recreate(self) -> SingletonThreadPool:
        self.logger.info('Pool recreating')
        return self.__class__(self._creator, pool_size=self.size, recycle=self._recycle, echo=self.echo, pre_ping=self._pre_ping, logging_name=self._orig_logging_name, reset_on_return=self._reset_on_return, _dispatch=self.dispatch, dialect=self._dialect)

    def dispose(self) -> None:
        """Dispose of this pool."""
        for conn in self._all_conns:
            try:
                conn.close()
            except Exception:
                pass
        self._all_conns.clear()

    def _cleanup(self) -> None:
        while len(self._all_conns) >= self.size:
            c = self._all_conns.pop()
            c.close()

    def status(self) -> str:
        return 'SingletonThreadPool id:%d size: %d' % (id(self), len(self._all_conns))

    def _do_return_conn(self, record: ConnectionPoolEntry) -> None:
        try:
            del self._fairy.current
        except AttributeError:
            pass

    def _do_get(self) -> ConnectionPoolEntry:
        try:
            if TYPE_CHECKING:
                c = cast(ConnectionPoolEntry, self._conn.current())
            else:
                c = self._conn.current()
            if c:
                return c
        except AttributeError:
            pass
        c = self._create_connection()
        self._conn.current = weakref.ref(c)
        if len(self._all_conns) >= self.size:
            self._cleanup()
        self._all_conns.add(c)
        return c

    def connect(self) -> PoolProxiedConnection:
        try:
            rec = cast(_ConnectionFairy, self._fairy.current())
        except AttributeError:
            pass
        else:
            if rec is not None:
                return rec._checkout_existing()
        return _ConnectionFairy._checkout(self, self._fairy)