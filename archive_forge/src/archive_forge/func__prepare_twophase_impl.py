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
def _prepare_twophase_impl(self, xid: Any) -> None:
    if self._has_events or self.engine._has_events:
        self.dispatch.prepare_twophase(self, xid)
    assert isinstance(self._transaction, TwoPhaseTransaction)
    try:
        self.engine.dialect.do_prepare_twophase(self, xid)
    except BaseException as e:
        self._handle_dbapi_exception(e, None, None, None, None)