from __future__ import annotations
from decimal import Decimal
from enum import IntEnum
import itertools
import operator
import re
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple as typing_Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from ._typing import has_schema_attr
from ._typing import is_named_from_clause
from ._typing import is_quoted_name
from ._typing import is_tuple_type
from .annotation import Annotated
from .annotation import SupportsWrappingAnnotations
from .base import _clone
from .base import _expand_cloned
from .base import _generative
from .base import _NoArg
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .base import Immutable
from .base import NO_ARG
from .base import SingletonConstant
from .cache_key import MemoizedHasCacheKey
from .cache_key import NO_CACHE
from .coercions import _document_text_coercion  # noqa
from .operators import ColumnOperators
from .traversals import HasCopyInternals
from .visitors import cloned_traverse
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .visitors import traverse
from .visitors import Visitable
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util import TypingOnly
from ..util.typing import Literal
from ..util.typing import Self
class ExpressionClauseList(OperatorExpression[_T]):
    """Describe a list of clauses, separated by an operator,
    in a column expression context.

    :class:`.ExpressionClauseList` differs from :class:`.ClauseList` in that
    it represents a column-oriented DQL expression only, not an open ended
    list of anything comma separated.

    .. versionadded:: 2.0

    """
    __visit_name__ = 'expression_clauselist'
    _traverse_internals: _TraverseInternalsType = [('clauses', InternalTraversal.dp_clauseelement_tuple), ('operator', InternalTraversal.dp_operator)]
    clauses: typing_Tuple[ColumnElement[Any], ...]
    group: bool

    def __init__(self, operator: OperatorType, *clauses: _ColumnExpressionArgument[Any], type_: Optional[_TypeEngineArgument[_T]]=None):
        self.operator = operator
        self.clauses = tuple((coercions.expect(roles.ExpressionElementRole, clause, apply_propagate_attrs=self) for clause in clauses))
        self._is_implicitly_boolean = operators.is_boolean(self.operator)
        self.type = type_api.to_instance(type_)

    @property
    def _flattened_operator_clauses(self) -> typing_Tuple[ColumnElement[Any], ...]:
        return self.clauses

    def __iter__(self) -> Iterator[ColumnElement[Any]]:
        return iter(self.clauses)

    def __len__(self) -> int:
        return len(self.clauses)

    @property
    def _select_iterable(self) -> _SelectIterable:
        return (self,)

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return list(itertools.chain(*[c._from_objects for c in self.clauses]))

    def _append_inplace(self, clause: ColumnElement[Any]) -> None:
        self.clauses += (clause,)

    @classmethod
    def _construct_for_list(cls, operator: OperatorType, type_: TypeEngine[_T], *clauses: ColumnElement[Any], group: bool=True) -> ExpressionClauseList[_T]:
        self = cls.__new__(cls)
        self.group = group
        if group:
            self.clauses = tuple((c.self_group(against=operator) for c in clauses))
        else:
            self.clauses = clauses
        self.operator = operator
        self.type = type_
        return self

    def _negate(self) -> Any:
        grouped = self.self_group(against=operators.inv)
        assert isinstance(grouped, ColumnElement)
        return UnaryExpression(grouped, operator=operators.inv, wraps_column_expression=True)