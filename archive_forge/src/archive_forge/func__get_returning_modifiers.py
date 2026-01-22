from __future__ import annotations
import functools
import operator
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import coercions
from . import dml
from . import elements
from . import roles
from .base import _DefaultDescriptionTuple
from .dml import isinsert as _compile_state_isinsert
from .elements import ColumnClause
from .schema import default_is_clause_element
from .schema import default_is_sequence
from .selectable import Select
from .selectable import TableClause
from .. import exc
from .. import util
from ..util.typing import Literal
def _get_returning_modifiers(compiler, stmt, compile_state, toplevel):
    """determines RETURNING strategy, if any, for the statement.

    This is where it's determined what we need to fetch from the
    INSERT or UPDATE statement after it's invoked.

    """
    dialect = compiler.dialect
    need_pks = toplevel and _compile_state_isinsert(compile_state) and (not stmt._inline) and (not compiler.for_executemany or (dialect.insert_executemany_returning and stmt._return_defaults)) and (not stmt._returning) and (not compile_state._has_multi_parameters)
    postfetch_lastrowid = need_pks and dialect.postfetch_lastrowid and (stmt.table._autoincrement_column is not None)
    implicit_returning = need_pks and dialect.insert_returning and compile_state._primary_table.implicit_returning and compile_state._supports_implicit_returning and ((not postfetch_lastrowid or dialect.favor_returning_over_lastrowid) or compile_state._has_multi_parameters or stmt._return_defaults)
    if implicit_returning:
        postfetch_lastrowid = False
    if _compile_state_isinsert(compile_state):
        should_implicit_return_defaults = implicit_returning and stmt._return_defaults
        explicit_returning = should_implicit_return_defaults or stmt._returning or stmt._supplemental_returning
        use_insertmanyvalues = toplevel and compiler.for_executemany and dialect.use_insertmanyvalues and (explicit_returning or dialect.use_insertmanyvalues_wo_returning)
        use_sentinel_columns = None
        if use_insertmanyvalues and explicit_returning and stmt._sort_by_parameter_order:
            use_sentinel_columns = compiler._get_sentinel_column_for_table(stmt.table)
    elif compile_state.isupdate:
        should_implicit_return_defaults = stmt._return_defaults and compile_state._primary_table.implicit_returning and compile_state._supports_implicit_returning and dialect.update_returning
        use_insertmanyvalues = False
        use_sentinel_columns = None
    elif compile_state.isdelete:
        should_implicit_return_defaults = stmt._return_defaults and compile_state._primary_table.implicit_returning and compile_state._supports_implicit_returning and dialect.delete_returning
        use_insertmanyvalues = False
        use_sentinel_columns = None
    else:
        should_implicit_return_defaults = False
        use_insertmanyvalues = False
        use_sentinel_columns = None
    if should_implicit_return_defaults:
        if not stmt._return_defaults_columns:
            implicit_return_defaults = set(stmt.table.c)
        else:
            implicit_return_defaults = set(stmt._return_defaults_columns)
    else:
        implicit_return_defaults = None
    return (need_pks, implicit_returning or should_implicit_return_defaults, implicit_return_defaults, postfetch_lastrowid, use_insertmanyvalues, use_sentinel_columns)