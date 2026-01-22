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
def _get_update_multitable_params(compiler, stmt, compile_state, stmt_parameter_tuples, check_columns, _col_bind_name, _getattr_col_key, values, kw):
    normalized_params = {coercions.expect(roles.DMLColumnRole, c): param for c, param in stmt_parameter_tuples or ()}
    include_table = compile_state.include_table_with_column_exprs
    affected_tables = set()
    for t in compile_state._extra_froms:
        for c in t.c:
            if c in normalized_params:
                affected_tables.add(t)
                check_columns[_getattr_col_key(c)] = c
                value = normalized_params[c]
                col_value = compiler.process(c, include_table=include_table)
                if coercions._is_literal(value):
                    value = _create_bind_param(compiler, c, value, required=value is REQUIRED, name=_col_bind_name(c), **kw)
                    accumulated_bind_names: Iterable[str] = (c.key,)
                elif value._is_bind_parameter:
                    cbn = _col_bind_name(c)
                    value = _handle_values_anonymous_param(compiler, c, value, name=cbn, **kw)
                    accumulated_bind_names = (cbn,)
                else:
                    compiler.postfetch.append(c)
                    value = compiler.process(value.self_group(), **kw)
                    accumulated_bind_names = ()
                values.append((c, col_value, value, accumulated_bind_names))
    for t in affected_tables:
        for c in t.c:
            if c in normalized_params:
                continue
            elif c.onupdate is not None and (not c.onupdate.is_sequence):
                if c.onupdate.is_clause_element:
                    values.append((c, compiler.process(c, include_table=include_table), compiler.process(c.onupdate.arg.self_group(), **kw), ()))
                    compiler.postfetch.append(c)
                else:
                    values.append((c, compiler.process(c, include_table=include_table), _create_update_prefetch_bind_param(compiler, c, name=_col_bind_name(c), **kw), (c.key,)))
            elif c.server_onupdate is not None:
                compiler.postfetch.append(c)