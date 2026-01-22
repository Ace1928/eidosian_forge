from __future__ import annotations
import asyncio
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import engine
from .base import ReversibleProxy
from .base import StartableContext
from .result import _ensure_sync_result
from .result import AsyncResult
from .result import AsyncScalarResult
from ... import util
from ...orm import close_all_sessions as _sync_close_all_sessions
from ...orm import object_session
from ...orm import Session
from ...orm import SessionTransaction
from ...orm import state as _instance_state
from ...util.concurrency import greenlet_spawn
from ...util.typing import Concatenate
from ...util.typing import ParamSpec
class AsyncSessionTransaction(ReversibleProxy[SessionTransaction], StartableContext['AsyncSessionTransaction']):
    """A wrapper for the ORM :class:`_orm.SessionTransaction` object.

    This object is provided so that a transaction-holding object
    for the :meth:`_asyncio.AsyncSession.begin` may be returned.

    The object supports both explicit calls to
    :meth:`_asyncio.AsyncSessionTransaction.commit` and
    :meth:`_asyncio.AsyncSessionTransaction.rollback`, as well as use as an
    async context manager.


    .. versionadded:: 1.4

    """
    __slots__ = ('session', 'sync_transaction', 'nested')
    session: AsyncSession
    sync_transaction: Optional[SessionTransaction]

    def __init__(self, session: AsyncSession, nested: bool=False):
        self.session = session
        self.nested = nested
        self.sync_transaction = None

    @property
    def is_active(self) -> bool:
        return self._sync_transaction() is not None and self._sync_transaction().is_active

    def _sync_transaction(self) -> SessionTransaction:
        if not self.sync_transaction:
            self._raise_for_not_started()
        return self.sync_transaction

    async def rollback(self) -> None:
        """Roll back this :class:`_asyncio.AsyncTransaction`."""
        await greenlet_spawn(self._sync_transaction().rollback)

    async def commit(self) -> None:
        """Commit this :class:`_asyncio.AsyncTransaction`."""
        await greenlet_spawn(self._sync_transaction().commit)

    async def start(self, is_ctxmanager: bool=False) -> AsyncSessionTransaction:
        self.sync_transaction = self._assign_proxied(await greenlet_spawn(self.session.sync_session.begin_nested if self.nested else self.session.sync_session.begin))
        if is_ctxmanager:
            self.sync_transaction.__enter__()
        return self

    async def __aexit__(self, type_: Any, value: Any, traceback: Any) -> None:
        await greenlet_spawn(self._sync_transaction().__exit__, type_, value, traceback)