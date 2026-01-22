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
def _append_param_parameter(compiler, stmt, compile_state, c, col_key, parameters, _col_bind_name, implicit_returning, implicit_return_defaults, postfetch_lastrowid, values, autoincrement_col, insert_null_pk_still_autoincrements, kw):
    value = parameters.pop(col_key)
    col_value = compiler.preparer.format_column(c, use_table=compile_state.include_table_with_column_exprs)
    accumulated_bind_names: Set[str] = set()
    if coercions._is_literal(value):
        if insert_null_pk_still_autoincrements and c.primary_key and (c is autoincrement_col):
            if postfetch_lastrowid:
                compiler.postfetch_lastrowid = True
            elif implicit_returning:
                compiler.implicit_returning.append(c)
        value = _create_bind_param(compiler, c, value, required=value is REQUIRED, name=_col_bind_name(c) if not _compile_state_isinsert(compile_state) or not compile_state._has_multi_parameters else '%s_m0' % _col_bind_name(c), accumulate_bind_names=accumulated_bind_names, **kw)
    elif value._is_bind_parameter:
        if insert_null_pk_still_autoincrements and value.value is None and c.primary_key and (c is autoincrement_col):
            if implicit_returning:
                compiler.implicit_returning.append(c)
            elif compiler.dialect.postfetch_lastrowid:
                compiler.postfetch_lastrowid = True
        value = _handle_values_anonymous_param(compiler, c, value, name=_col_bind_name(c) if not _compile_state_isinsert(compile_state) or not compile_state._has_multi_parameters else '%s_m0' % _col_bind_name(c), accumulate_bind_names=accumulated_bind_names, **kw)
    else:
        value = compiler.process(value.self_group(), accumulate_bind_names=accumulated_bind_names, **kw)
        if compile_state.isupdate:
            if implicit_return_defaults and c in implicit_return_defaults:
                compiler.implicit_returning.append(c)
            else:
                compiler.postfetch.append(c)
        elif c.primary_key:
            if implicit_returning:
                compiler.implicit_returning.append(c)
            elif compiler.dialect.postfetch_lastrowid:
                compiler.postfetch_lastrowid = True
        elif implicit_return_defaults and c in implicit_return_defaults:
            compiler.implicit_returning.append(c)
        else:
            compiler.postfetch.append(c)
    values.append((c, col_value, value, accumulated_bind_names))