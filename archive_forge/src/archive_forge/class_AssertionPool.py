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
class AssertionPool(Pool):
    """A :class:`_pool.Pool` that allows at most one checked out connection at
    any given time.

    This will raise an exception if more than one connection is checked out
    at a time.  Useful for debugging code that is using more connections
    than desired.

    The :class:`.AssertionPool` class **is compatible** with asyncio and
    :func:`_asyncio.create_async_engine`.

    """
    _conn: Optional[ConnectionPoolEntry]
    _checkout_traceback: Optional[List[str]]

    def __init__(self, *args: Any, **kw: Any):
        self._conn = None
        self._checked_out = False
        self._store_traceback = kw.pop('store_traceback', True)
        self._checkout_traceback = None
        Pool.__init__(self, *args, **kw)

    def status(self) -> str:
        return 'AssertionPool'

    def _do_return_conn(self, record: ConnectionPoolEntry) -> None:
        if not self._checked_out:
            raise AssertionError('connection is not checked out')
        self._checked_out = False
        assert record is self._conn

    def dispose(self) -> None:
        self._checked_out = False
        if self._conn:
            self._conn.close()

    def recreate(self) -> AssertionPool:
        self.logger.info('Pool recreating')
        return self.__class__(self._creator, echo=self.echo, pre_ping=self._pre_ping, recycle=self._recycle, reset_on_return=self._reset_on_return, logging_name=self._orig_logging_name, _dispatch=self.dispatch, dialect=self._dialect)

    def _do_get(self) -> ConnectionPoolEntry:
        if self._checked_out:
            if self._checkout_traceback:
                suffix = ' at:\n%s' % ''.join(chop_traceback(self._checkout_traceback))
            else:
                suffix = ''
            raise AssertionError('connection is already checked out' + suffix)
        if not self._conn:
            self._conn = self._create_connection()
        self._checked_out = True
        if self._store_traceback:
            self._checkout_traceback = traceback.format_stack()
        return self._conn