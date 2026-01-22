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
class OptionEngineMixin(log.Identified):
    _sa_propagate_class_events = False
    dispatch: dispatcher[ConnectionEventsTarget]
    _compiled_cache: Optional[CompiledCacheType]
    dialect: Dialect
    pool: Pool
    url: URL
    hide_parameters: bool
    echo: log.echo_property

    def __init__(self, proxied: Engine, execution_options: CoreExecuteOptionsParameter):
        self._proxied = proxied
        self.url = proxied.url
        self.dialect = proxied.dialect
        self.logging_name = proxied.logging_name
        self.echo = proxied.echo
        self._compiled_cache = proxied._compiled_cache
        self.hide_parameters = proxied.hide_parameters
        log.instance_logger(self, echoflag=self.echo)
        self.dispatch = self.dispatch._join(proxied.dispatch)
        self._execution_options = proxied._execution_options
        self.update_execution_options(**execution_options)

    def update_execution_options(self, **opt: Any) -> None:
        raise NotImplementedError()
    if not typing.TYPE_CHECKING:

        @property
        def pool(self) -> Pool:
            return self._proxied.pool

        @pool.setter
        def pool(self, pool: Pool) -> None:
            self._proxied.pool = pool

        @property
        def _has_events(self) -> bool:
            return self._proxied._has_events or self.__dict__.get('_has_events', False)

        @_has_events.setter
        def _has_events(self, value: bool) -> None:
            self.__dict__['_has_events'] = value