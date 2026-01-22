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
def _setup_for_bulk_update(self, statement, compiler, **kw):
    """establish an UPDATE statement within the context of
        bulk insert.

        This method will be within the "conn.execute()" call that is invoked
        by persistence._emit_update_statement().

        """
    statement = cast(dml.Update, statement)
    an = statement._annotations
    emit_update_table, _ = (an['_emit_update_table'], an['_emit_update_mapper'])
    statement = statement._clone()
    statement.table = emit_update_table
    UpdateDMLState.__init__(self, statement, compiler, **kw)
    if self._ordered_values:
        raise sa_exc.InvalidRequestError('bulk ORM UPDATE does not support ordered_values() for custom UPDATE statements with bulk parameter sets.  Use a non-bulk UPDATE statement or use values().')
    if self._dict_parameters:
        self._dict_parameters = {col: val for col, val in self._dict_parameters.items() if col.table is emit_update_table}
    self.statement = statement