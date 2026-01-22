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
class ClauseList(roles.InElementRole, roles.OrderByRole, roles.ColumnsClauseRole, roles.DMLColumnRole, DQLDMLClauseElement):
    """Describe a list of clauses, separated by an operator.

    By default, is comma-separated, such as a column listing.

    """
    __visit_name__ = 'clauselist'
    _is_clause_list = True
    _traverse_internals: _TraverseInternalsType = [('clauses', InternalTraversal.dp_clauseelement_list), ('operator', InternalTraversal.dp_operator)]
    clauses: List[ColumnElement[Any]]

    def __init__(self, *clauses: _ColumnExpressionArgument[Any], operator: OperatorType=operators.comma_op, group: bool=True, group_contents: bool=True, _literal_as_text_role: Type[roles.SQLRole]=roles.WhereHavingRole):
        self.operator = operator
        self.group = group
        self.group_contents = group_contents
        clauses_iterator: Iterable[_ColumnExpressionArgument[Any]] = clauses
        text_converter_role: Type[roles.SQLRole] = _literal_as_text_role
        self._text_converter_role = text_converter_role
        if self.group_contents:
            self.clauses = [coercions.expect(text_converter_role, clause, apply_propagate_attrs=self).self_group(against=self.operator) for clause in clauses_iterator]
        else:
            self.clauses = [coercions.expect(text_converter_role, clause, apply_propagate_attrs=self) for clause in clauses_iterator]
        self._is_implicitly_boolean = operators.is_boolean(self.operator)

    @classmethod
    def _construct_raw(cls, operator: OperatorType, clauses: Optional[Sequence[ColumnElement[Any]]]=None) -> ClauseList:
        self = cls.__new__(cls)
        self.clauses = list(clauses) if clauses else []
        self.group = True
        self.operator = operator
        self.group_contents = True
        self._is_implicitly_boolean = False
        return self

    def __iter__(self) -> Iterator[ColumnElement[Any]]:
        return iter(self.clauses)

    def __len__(self) -> int:
        return len(self.clauses)

    @property
    def _select_iterable(self) -> _SelectIterable:
        return itertools.chain.from_iterable([elem._select_iterable for elem in self.clauses])

    def append(self, clause):
        if self.group_contents:
            self.clauses.append(coercions.expect(self._text_converter_role, clause).self_group(against=self.operator))
        else:
            self.clauses.append(coercions.expect(self._text_converter_role, clause))

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return list(itertools.chain(*[c._from_objects for c in self.clauses]))

    def self_group(self, against=None):
        if self.group and operators.is_precedent(self.operator, against):
            return Grouping(self)
        else:
            return self