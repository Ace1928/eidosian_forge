from __future__ import annotations
import contextlib
from enum import Enum
import itertools
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import bulk_persistence
from . import context
from . import descriptor_props
from . import exc
from . import identity
from . import loading
from . import query
from . import state as statelib
from ._typing import _O
from ._typing import insp_is_mapper
from ._typing import is_composite_class
from ._typing import is_orm_option
from ._typing import is_user_defined_option
from .base import _class_to_mapper
from .base import _none_set
from .base import _state_mapper
from .base import instance_str
from .base import LoaderCallableStatus
from .base import object_mapper
from .base import object_state
from .base import PassiveFlag
from .base import state_str
from .context import FromStatement
from .context import ORMCompileState
from .identity import IdentityMap
from .query import Query
from .state import InstanceState
from .state_changes import _StateChange
from .state_changes import _StateChangeState
from .state_changes import _StateChangeStates
from .unitofwork import UOWTransaction
from .. import engine
from .. import exc as sa_exc
from .. import sql
from .. import util
from ..engine import Connection
from ..engine import Engine
from ..engine.util import TransactionalContext
from ..event import dispatcher
from ..event import EventTarget
from ..inspection import inspect
from ..inspection import Inspectable
from ..sql import coercions
from ..sql import dml
from ..sql import roles
from ..sql import Select
from ..sql import TableClause
from ..sql import visitors
from ..sql.base import _NoArg
from ..sql.base import CompileState
from ..sql.schema import Table
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import IdentitySet
from ..util.typing import Literal
from ..util.typing import Protocol
def _execute_internal(self, statement: Executable, params: Optional[_CoreAnyExecuteParams]=None, *, execution_options: OrmExecuteOptionsParameter=util.EMPTY_DICT, bind_arguments: Optional[_BindArguments]=None, _parent_execute_state: Optional[Any]=None, _add_event: Optional[Any]=None, _scalar_result: bool=False) -> Any:
    statement = coercions.expect(roles.StatementRole, statement)
    if not bind_arguments:
        bind_arguments = {}
    else:
        bind_arguments = dict(bind_arguments)
    if statement._propagate_attrs.get('compile_state_plugin', None) == 'orm':
        compile_state_cls = CompileState._get_plugin_class_for_plugin(statement, 'orm')
        if TYPE_CHECKING:
            assert isinstance(compile_state_cls, context.AbstractORMCompileState)
    else:
        compile_state_cls = None
        bind_arguments.setdefault('clause', statement)
    execution_options = util.coerce_to_immutabledict(execution_options)
    if _parent_execute_state:
        events_todo = _parent_execute_state._remaining_events()
    else:
        events_todo = self.dispatch.do_orm_execute
        if _add_event:
            events_todo = list(events_todo) + [_add_event]
    if events_todo:
        if compile_state_cls is not None:
            statement, execution_options = compile_state_cls.orm_pre_session_exec(self, statement, params, execution_options, bind_arguments, True)
        orm_exec_state = ORMExecuteState(self, statement, params, execution_options, bind_arguments, compile_state_cls, events_todo)
        for idx, fn in enumerate(events_todo):
            orm_exec_state._starting_event_idx = idx
            fn_result: Optional[Result[Any]] = fn(orm_exec_state)
            if fn_result:
                if _scalar_result:
                    return fn_result.scalar()
                else:
                    return fn_result
        statement = orm_exec_state.statement
        execution_options = orm_exec_state.local_execution_options
    if compile_state_cls is not None:
        statement, execution_options = compile_state_cls.orm_pre_session_exec(self, statement, params, execution_options, bind_arguments, False)
    bind = self.get_bind(**bind_arguments)
    conn = self._connection_for_bind(bind)
    if _scalar_result and (not compile_state_cls):
        if TYPE_CHECKING:
            params = cast(_CoreSingleExecuteParams, params)
        return conn.scalar(statement, params or {}, execution_options=execution_options)
    if compile_state_cls:
        result: Result[Any] = compile_state_cls.orm_execute_statement(self, statement, params or {}, execution_options, bind_arguments, conn)
    else:
        result = conn.execute(statement, params or {}, execution_options=execution_options)
    if _scalar_result:
        return result.scalar()
    else:
        return result