from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import context
from . import evaluator
from . import exc as orm_exc
from . import loading
from . import persistence
from .base import NO_VALUE
from .context import AbstractORMCompileState
from .context import FromStatement
from .context import ORMFromStatementCompileState
from .context import QueryContext
from .. import exc as sa_exc
from .. import util
from ..engine import Dialect
from ..engine import result as _result
from ..sql import coercions
from ..sql import dml
from ..sql import expression
from ..sql import roles
from ..sql import select
from ..sql import sqltypes
from ..sql.base import _entity_namespace_key
from ..sql.base import CompileState
from ..sql.base import Options
from ..sql.dml import DeleteDMLState
from ..sql.dml import InsertDMLState
from ..sql.dml import UpdateDMLState
from ..util import EMPTY_DICT
from ..util.typing import Literal
@classmethod
def _return_orm_returning(cls, session, statement, params, execution_options, bind_arguments, result):
    execution_context = result.context
    compile_state = execution_context.compiled.compile_state
    if compile_state.from_statement_ctx and (not compile_state.from_statement_ctx.compile_options._is_star):
        load_options = execution_options.get('_sa_orm_load_options', QueryContext.default_load_options)
        querycontext = QueryContext(compile_state.from_statement_ctx, compile_state.select_statement, params, session, load_options, execution_options, bind_arguments)
        return loading.instances(result, querycontext)
    else:
        return result