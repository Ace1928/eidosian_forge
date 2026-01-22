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
def _append_param_update(compiler, compile_state, stmt, c, implicit_return_defaults, values, kw):
    include_table = compile_state.include_table_with_column_exprs
    if c.onupdate is not None and (not c.onupdate.is_sequence):
        if c.onupdate.is_clause_element:
            values.append((c, compiler.preparer.format_column(c, use_table=include_table), compiler.process(c.onupdate.arg.self_group(), **kw), ()))
            if implicit_return_defaults and c in implicit_return_defaults:
                compiler.implicit_returning.append(c)
            else:
                compiler.postfetch.append(c)
        else:
            values.append((c, compiler.preparer.format_column(c, use_table=include_table), _create_update_prefetch_bind_param(compiler, c, **kw), (c.key,)))
    elif c.server_onupdate is not None:
        if implicit_return_defaults and c in implicit_return_defaults:
            compiler.implicit_returning.append(c)
        else:
            compiler.postfetch.append(c)
    elif implicit_return_defaults and (stmt._return_defaults_columns or not stmt._return_defaults) and (c in implicit_return_defaults):
        compiler.implicit_returning.append(c)