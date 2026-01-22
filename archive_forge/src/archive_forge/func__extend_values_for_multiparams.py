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
def _extend_values_for_multiparams(compiler: SQLCompiler, stmt: ValuesBase, compile_state: DMLState, initial_values: Sequence[_CrudParamElementStr], _column_as_key: Callable[..., str], kw: Dict[str, Any]) -> List[Sequence[_CrudParamElementStr]]:
    values_0 = initial_values
    values = [initial_values]
    mp = compile_state._multi_parameters
    assert mp is not None
    for i, row in enumerate(mp[1:]):
        extension: List[_CrudParamElementStr] = []
        row = {_column_as_key(key): v for key, v in row.items()}
        for col, col_expr, param, accumulated_names in values_0:
            if col.key in row:
                key = col.key
                if coercions._is_literal(row[key]):
                    new_param = _create_bind_param(compiler, col, row[key], name='%s_m%d' % (col.key, i + 1), **kw)
                else:
                    new_param = compiler.process(row[key].self_group(), **kw)
            else:
                new_param = _process_multiparam_default_bind(compiler, stmt, col, i, kw)
            extension.append((col, col_expr, new_param, accumulated_names))
        values.append(extension)
    return values