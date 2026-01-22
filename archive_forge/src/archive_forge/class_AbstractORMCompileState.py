from __future__ import annotations
import itertools
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import interfaces
from . import loading
from .base import _is_aliased_class
from .interfaces import ORMColumnDescription
from .interfaces import ORMColumnsClauseRole
from .path_registry import PathRegistry
from .util import _entity_corresponds_to
from .util import _ORMJoin
from .util import _TraceAdaptRole
from .util import AliasedClass
from .util import Bundle
from .util import ORMAdapter
from .util import ORMStatementAdapter
from .. import exc as sa_exc
from .. import future
from .. import inspect
from .. import sql
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import _TP
from ..sql._typing import is_dml
from ..sql._typing import is_insert_update
from ..sql._typing import is_select_base
from ..sql.base import _select_iterables
from ..sql.base import CacheableOptions
from ..sql.base import CompileState
from ..sql.base import Executable
from ..sql.base import Generative
from ..sql.base import Options
from ..sql.dml import UpdateBase
from ..sql.elements import GroupedElement
from ..sql.elements import TextClause
from ..sql.selectable import CompoundSelectState
from ..sql.selectable import LABEL_STYLE_DISAMBIGUATE_ONLY
from ..sql.selectable import LABEL_STYLE_NONE
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import Select
from ..sql.selectable import SelectLabelStyle
from ..sql.selectable import SelectState
from ..sql.selectable import TypedReturnsRows
from ..sql.visitors import InternalTraversal
class AbstractORMCompileState(CompileState):
    is_dml_returning = False

    def _init_global_attributes(self, statement, compiler, *, toplevel, process_criteria_for_toplevel):
        self.attributes = {}
        if compiler is None:
            self.global_attributes = ga = {}
            assert toplevel
            return
        else:
            self.global_attributes = ga = compiler._global_attributes
        if toplevel:
            ga['toplevel_orm'] = True
            if process_criteria_for_toplevel:
                for opt in statement._with_options:
                    if opt._is_criteria_option:
                        opt.process_compile_state(self)
            return
        elif ga.get('toplevel_orm', False):
            return
        stack_0 = compiler.stack[0]
        try:
            toplevel_stmt = stack_0['selectable']
        except KeyError:
            pass
        else:
            for opt in toplevel_stmt._with_options:
                if opt._is_compile_state and opt._is_criteria_option:
                    opt.process_compile_state(self)
        ga['toplevel_orm'] = True

    @classmethod
    def create_for_statement(cls, statement: Union[Select, FromStatement], compiler: Optional[SQLCompiler], **kw: Any) -> AbstractORMCompileState:
        """Create a context for a statement given a :class:`.Compiler`.

        This method is always invoked in the context of SQLCompiler.process().

        For a Select object, this would be invoked from
        SQLCompiler.visit_select(). For the special FromStatement object used
        by Query to indicate "Query.from_statement()", this is called by
        FromStatement._compiler_dispatch() that would be called by
        SQLCompiler.process().
        """
        return super().create_for_statement(statement, compiler, **kw)

    @classmethod
    def orm_pre_session_exec(cls, session, statement, params, execution_options, bind_arguments, is_pre_event):
        raise NotImplementedError()

    @classmethod
    def orm_execute_statement(cls, session, statement, params, execution_options, bind_arguments, conn) -> Result:
        result = conn.execute(statement, params or {}, execution_options=execution_options)
        return cls.orm_setup_cursor_result(session, statement, params, execution_options, bind_arguments, result)

    @classmethod
    def orm_setup_cursor_result(cls, session, statement, params, execution_options, bind_arguments, result):
        raise NotImplementedError()