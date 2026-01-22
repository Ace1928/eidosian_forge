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
def _append_param_insert_pk_no_returning(compiler, stmt, c, values, kw):
    """Create a primary key expression in the INSERT statement where
    we want to populate result.inserted_primary_key and we cannot use
    RETURNING.

    Depending on the kind of default here we may create a bound parameter
    in the INSERT statement and pre-execute a default generation function,
    or we may use cursor.lastrowid if supported by the dialect.


    """
    if c.default is not None and (not c.default.is_sequence or (compiler.dialect.supports_sequences and (not c.default.optional or not compiler.dialect.sequences_optional))) or (c is stmt.table._autoincrement_column and (not compiler.dialect.postfetch_lastrowid and (c.default is not None and c.default.is_sequence and compiler.dialect.supports_sequences or (c.default is None and compiler.dialect.preexecute_autoincrement_sequences)))):
        values.append((c, compiler.preparer.format_column(c), _create_insert_prefetch_bind_param(compiler, c, **kw), (c.key,)))
    elif c.default is None and c.server_default is None and (not c.nullable) and (c is not stmt.table._autoincrement_column):
        _warn_pk_with_no_anticipated_value(c)
    elif compiler.dialect.postfetch_lastrowid:
        compiler.postfetch_lastrowid = True