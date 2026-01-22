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
class NestedTransaction(Transaction):
    """Represent a 'nested', or SAVEPOINT transaction.

    The :class:`.NestedTransaction` object is created by calling the
    :meth:`_engine.Connection.begin_nested` method of
    :class:`_engine.Connection`.

    When using :class:`.NestedTransaction`, the semantics of "begin" /
    "commit" / "rollback" are as follows:

    * the "begin" operation corresponds to the "BEGIN SAVEPOINT" command, where
      the savepoint is given an explicit name that is part of the state
      of this object.

    * The :meth:`.NestedTransaction.commit` method corresponds to a
      "RELEASE SAVEPOINT" operation, using the savepoint identifier associated
      with this :class:`.NestedTransaction`.

    * The :meth:`.NestedTransaction.rollback` method corresponds to a
      "ROLLBACK TO SAVEPOINT" operation, using the savepoint identifier
      associated with this :class:`.NestedTransaction`.

    The rationale for mimicking the semantics of an outer transaction in
    terms of savepoints so that code may deal with a "savepoint" transaction
    and an "outer" transaction in an agnostic way.

    .. seealso::

        :ref:`session_begin_nested` - ORM version of the SAVEPOINT API.

    """
    __slots__ = ('connection', 'is_active', '_savepoint', '_previous_nested')
    _savepoint: str

    def __init__(self, connection: Connection):
        assert connection._transaction is not None
        if connection._trans_context_manager:
            TransactionalContext._trans_ctx_check(connection)
        self.connection = connection
        self._savepoint = self.connection._savepoint_impl()
        self.is_active = True
        self._previous_nested = connection._nested_transaction
        connection._nested_transaction = self

    def _deactivate_from_connection(self, warn: bool=True) -> None:
        if self.connection._nested_transaction is self:
            self.connection._nested_transaction = self._previous_nested
        elif warn:
            util.warn('nested transaction already deassociated from connection')

    @property
    def _deactivated_from_connection(self) -> bool:
        return self.connection._nested_transaction is not self

    def _cancel(self) -> None:
        self.is_active = False
        self._deactivate_from_connection()
        if self._previous_nested:
            self._previous_nested._cancel()

    def _close_impl(self, deactivate_from_connection: bool, warn_already_deactive: bool) -> None:
        try:
            if self.is_active and self.connection._transaction and self.connection._transaction.is_active:
                self.connection._rollback_to_savepoint_impl(self._savepoint)
        finally:
            self.is_active = False
            if deactivate_from_connection:
                self._deactivate_from_connection(warn=warn_already_deactive)
        assert not self.is_active
        if deactivate_from_connection:
            assert self.connection._nested_transaction is not self

    def _do_close(self) -> None:
        self._close_impl(True, False)

    def _do_rollback(self) -> None:
        self._close_impl(True, True)

    def _do_commit(self) -> None:
        if self.is_active:
            try:
                self.connection._release_savepoint_impl(self._savepoint)
            finally:
                self.is_active = False
            self._deactivate_from_connection()
        elif self.connection._nested_transaction is self:
            self.connection._invalid_transaction()
        else:
            raise exc.InvalidRequestError('This nested transaction is inactive')