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
def _append_param_insert_pk_returning(compiler, stmt, c, values, kw):
    """Create a primary key expression in the INSERT statement where
    we want to populate result.inserted_primary_key and RETURNING
    is available.

    """
    if c.default is not None:
        if c.default.is_sequence:
            if compiler.dialect.supports_sequences and (not c.default.optional or not compiler.dialect.sequences_optional):
                accumulated_bind_names: Set[str] = set()
                values.append((c, compiler.preparer.format_column(c), compiler.process(c.default, accumulate_bind_names=accumulated_bind_names, **kw), accumulated_bind_names))
            compiler.implicit_returning.append(c)
        elif c.default.is_clause_element:
            accumulated_bind_names = set()
            values.append((c, compiler.preparer.format_column(c), compiler.process(c.default.arg.self_group(), accumulate_bind_names=accumulated_bind_names, **kw), accumulated_bind_names))
            compiler.implicit_returning.append(c)
        else:
            values.append((c, compiler.preparer.format_column(c), _create_insert_prefetch_bind_param(compiler, c, **kw), (c.key,)))
    elif c is stmt.table._autoincrement_column or c.server_default is not None:
        compiler.implicit_returning.append(c)
    elif not c.nullable:
        _warn_pk_with_no_anticipated_value(c)