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
def _setup_orm_returning(self, compiler, orm_level_statement, dml_level_statement, dml_mapper, *, use_supplemental_cols=True):
    """establish ORM column handlers for an INSERT, UPDATE, or DELETE
        which uses explicit returning().

        called within compilation level create_for_statement.

        The _return_orm_returning() method then receives the Result
        after the statement was executed, and applies ORM loading to the
        state that we first established here.

        """
    if orm_level_statement._returning:
        fs = FromStatement(orm_level_statement._returning, dml_level_statement, _adapt_on_names=False)
        fs = fs.execution_options(**orm_level_statement._execution_options)
        fs = fs.options(*orm_level_statement._with_options)
        self.select_statement = fs
        self.from_statement_ctx = fsc = ORMFromStatementCompileState.create_for_statement(fs, compiler)
        fsc.setup_dml_returning_compile_state(dml_mapper)
        dml_level_statement = dml_level_statement._generate()
        dml_level_statement._returning = ()
        cols_to_return = [c for c in fsc.primary_columns if c is not None]
        if not cols_to_return:
            cols_to_return.extend(dml_mapper.primary_key)
        if use_supplemental_cols:
            dml_level_statement = dml_level_statement.return_defaults(*dml_mapper.primary_key, supplemental_cols=cols_to_return)
        else:
            dml_level_statement = dml_level_statement.returning(*cols_to_return)
    return dml_level_statement