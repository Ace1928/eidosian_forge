from __future__ import annotations
import datetime
import decimal
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import annotation
from . import coercions
from . import operators
from . import roles
from . import schema
from . import sqltypes
from . import type_api
from . import util as sqlutil
from ._typing import is_table_value_type
from .base import _entity_namespace
from .base import ColumnCollection
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .elements import _type_from_args
from .elements import BinaryExpression
from .elements import BindParameter
from .elements import Cast
from .elements import ClauseList
from .elements import ColumnElement
from .elements import Extract
from .elements import FunctionFilter
from .elements import Grouping
from .elements import literal_column
from .elements import NamedColumn
from .elements import Over
from .elements import WithinGroup
from .selectable import FromClause
from .selectable import Select
from .selectable import TableValuedAlias
from .sqltypes import TableValueType
from .type_api import TypeEngine
from .visitors import InternalTraversal
from .. import util
class FunctionAsBinary(BinaryExpression[Any]):
    _traverse_internals = [('sql_function', InternalTraversal.dp_clauseelement), ('left_index', InternalTraversal.dp_plain_obj), ('right_index', InternalTraversal.dp_plain_obj), ('modifiers', InternalTraversal.dp_plain_dict)]
    sql_function: FunctionElement[Any]
    left_index: int
    right_index: int

    def _gen_cache_key(self, anon_map: Any, bindparams: Any) -> Any:
        return ColumnElement._gen_cache_key(self, anon_map, bindparams)

    def __init__(self, fn: FunctionElement[Any], left_index: int, right_index: int):
        self.sql_function = fn
        self.left_index = left_index
        self.right_index = right_index
        self.operator = operators.function_as_comparison_op
        self.type = sqltypes.BOOLEANTYPE
        self.negate = None
        self._is_implicitly_boolean = True
        self.modifiers = {}

    @property
    def left_expr(self) -> ColumnElement[Any]:
        return self.sql_function.clauses.clauses[self.left_index - 1]

    @left_expr.setter
    def left_expr(self, value: ColumnElement[Any]) -> None:
        self.sql_function.clauses.clauses[self.left_index - 1] = value

    @property
    def right_expr(self) -> ColumnElement[Any]:
        return self.sql_function.clauses.clauses[self.right_index - 1]

    @right_expr.setter
    def right_expr(self, value: ColumnElement[Any]) -> None:
        self.sql_function.clauses.clauses[self.right_index - 1] = value
    if not TYPE_CHECKING:
        left = left_expr
        right = right_expr