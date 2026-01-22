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
def _cursor_execute(self, cursor: DBAPICursor, statement: str, parameters: _DBAPISingleExecuteParams, context: Optional[ExecutionContext]=None) -> None:
    """Execute a statement + params on the given cursor.

        Adds appropriate logging and exception handling.

        This method is used by DefaultDialect for special-case
        executions, such as for sequences and column defaults.
        The path of statement execution in the majority of cases
        terminates at _execute_context().

        """
    if self._has_events or self.engine._has_events:
        for fn in self.dispatch.before_cursor_execute:
            statement, parameters = fn(self, cursor, statement, parameters, context, False)
    if self._echo:
        self._log_info(statement)
        self._log_info('[raw sql] %r', parameters)
    try:
        for fn in () if not self.dialect._has_events else self.dialect.dispatch.do_execute:
            if fn(cursor, statement, parameters, context):
                break
        else:
            self.dialect.do_execute(cursor, statement, parameters, context)
    except BaseException as e:
        self._handle_dbapi_exception(e, statement, parameters, cursor, context)
    if self._has_events or self.engine._has_events:
        self.dispatch.after_cursor_execute(self, cursor, statement, parameters, context, False)