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
def _execute_default(self, default: DefaultGenerator, distilled_parameters: _CoreMultiExecuteParams, execution_options: CoreExecuteOptionsParameter) -> Any:
    """Execute a schema.ColumnDefault object."""
    execution_options = self._execution_options.merge_with(execution_options)
    event_multiparams: Optional[_CoreMultiExecuteParams]
    event_params: Optional[_CoreAnyExecuteParams]
    if self._has_events or self.engine._has_events:
        default, distilled_parameters, event_multiparams, event_params = self._invoke_before_exec_event(default, distilled_parameters, execution_options)
    else:
        event_multiparams = event_params = None
    try:
        conn = self._dbapi_connection
        if conn is None:
            conn = self._revalidate_connection()
        dialect = self.dialect
        ctx = dialect.execution_ctx_cls._init_default(dialect, self, conn, execution_options)
    except (exc.PendingRollbackError, exc.ResourceClosedError):
        raise
    except BaseException as e:
        self._handle_dbapi_exception(e, None, None, None, None)
    ret = ctx._exec_default(None, default, None)
    if self._has_events or self.engine._has_events:
        self.dispatch.after_execute(self, default, event_multiparams, event_params, execution_options, ret)
    return ret