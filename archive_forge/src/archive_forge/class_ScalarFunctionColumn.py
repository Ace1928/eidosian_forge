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
class ScalarFunctionColumn(NamedColumn[_T]):
    __visit_name__ = 'scalar_function_column'
    _traverse_internals = [('name', InternalTraversal.dp_anon_name), ('type', InternalTraversal.dp_type), ('fn', InternalTraversal.dp_clauseelement)]
    is_literal = False
    table = None

    def __init__(self, fn: FunctionElement[_T], name: str, type_: Optional[_TypeEngineArgument[_T]]=None):
        self.fn = fn
        self.name = name
        self.type = type_api.to_instance(type_)