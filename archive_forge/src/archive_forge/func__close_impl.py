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