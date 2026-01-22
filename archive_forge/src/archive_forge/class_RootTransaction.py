from __future__ import annotations
import contextlib
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from .interfaces import BindTyping
from .interfaces import ConnectionEventsTarget
from .interfaces import DBAPICursor
from .interfaces import ExceptionContext
from .interfaces import ExecuteStyle
from .interfaces import ExecutionContext
from .interfaces import IsolationLevel
from .util import _distill_params_20
from .util import _distill_raw_params
from .util import TransactionalContext
from .. import exc
from .. import inspection
from .. import log
from .. import util
from ..sql import compiler
from ..sql import util as sql_util
class RootTransaction(Transaction):
    """Represent the "root" transaction on a :class:`_engine.Connection`.

    This corresponds to the current "BEGIN/COMMIT/ROLLBACK" that's occurring
    for the :class:`_engine.Connection`. The :class:`_engine.RootTransaction`
    is created by calling upon the :meth:`_engine.Connection.begin` method, and
    remains associated with the :class:`_engine.Connection` throughout its
    active span. The current :class:`_engine.RootTransaction` in use is
    accessible via the :attr:`_engine.Connection.get_transaction` method of
    :class:`_engine.Connection`.

    In :term:`2.0 style` use, the :class:`_engine.Connection` also employs
    "autobegin" behavior that will create a new
    :class:`_engine.RootTransaction` whenever a connection in a
    non-transactional state is used to emit commands on the DBAPI connection.
    The scope of the :class:`_engine.RootTransaction` in 2.0 style
    use can be controlled using the :meth:`_engine.Connection.commit` and
    :meth:`_engine.Connection.rollback` methods.


    """
    _is_root = True
    __slots__ = ('connection', 'is_active')

    def __init__(self, connection: Connection):
        assert connection._transaction is None
        if connection._trans_context_manager:
            TransactionalContext._trans_ctx_check(connection)
        self.connection = connection
        self._connection_begin_impl()
        connection._transaction = self
        self.is_active = True

    def _deactivate_from_connection(self) -> None:
        if self.is_active:
            assert self.connection._transaction is self
            self.is_active = False
        elif self.connection._transaction is not self:
            util.warn('transaction already deassociated from connection')

    @property
    def _deactivated_from_connection(self) -> bool:
        return self.connection._transaction is not self

    def _connection_begin_impl(self) -> None:
        self.connection._begin_impl(self)

    def _connection_rollback_impl(self) -> None:
        self.connection._rollback_impl()

    def _connection_commit_impl(self) -> None:
        self.connection._commit_impl()

    def _close_impl(self, try_deactivate: bool=False) -> None:
        try:
            if self.is_active:
                self._connection_rollback_impl()
            if self.connection._nested_transaction:
                self.connection._nested_transaction._cancel()
        finally:
            if self.is_active or try_deactivate:
                self._deactivate_from_connection()
            if self.connection._transaction is self:
                self.connection._transaction = None
        assert not self.is_active
        assert self.connection._transaction is not self

    def _do_close(self) -> None:
        self._close_impl()

    def _do_rollback(self) -> None:
        self._close_impl(try_deactivate=True)

    def _do_commit(self) -> None:
        if self.is_active:
            assert self.connection._transaction is self
            try:
                self._connection_commit_impl()
            finally:
                if self.connection._nested_transaction:
                    self.connection._nested_transaction._cancel()
                self._deactivate_from_connection()
            self.connection._transaction = None
        elif self.connection._transaction is self:
            self.connection._invalid_transaction()
        else:
            raise exc.InvalidRequestError('This transaction is inactive')
        assert not self.is_active
        assert self.connection._transaction is not self