from __future__ import annotations
import collections
from enum import Enum
import itertools
from typing import AbstractSet
from typing import Any as TODO_Any
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import cache_key
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from . import visitors
from ._typing import _ColumnsClauseArgument
from ._typing import _no_kw
from ._typing import _TP
from ._typing import is_column_element
from ._typing import is_select_statement
from ._typing import is_subquery
from ._typing import is_table
from ._typing import is_text_clause
from .annotation import Annotated
from .annotation import SupportsCloneAnnotations
from .base import _clone
from .base import _cloned_difference
from .base import _cloned_intersection
from .base import _entity_namespace_key
from .base import _EntityNamespace
from .base import _expand_cloned
from .base import _from_objects
from .base import _generative
from .base import _never_select_column
from .base import _NoArg
from .base import _select_iterables
from .base import CacheableOptions
from .base import ColumnCollection
from .base import ColumnSet
from .base import CompileState
from .base import DedupeColumnCollection
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .base import HasMemoized
from .base import Immutable
from .coercions import _document_text_coercion
from .elements import _anonymous_label
from .elements import BindParameter
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ClauseList
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import DQLDMLClauseElement
from .elements import GroupedElement
from .elements import literal_column
from .elements import TableValuedColumn
from .elements import UnaryExpression
from .operators import OperatorType
from .sqltypes import NULLTYPE
from .visitors import _TraverseInternalsType
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import exc
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
class FromGrouping(GroupedElement, FromClause):
    """Represent a grouping of a FROM clause"""
    _traverse_internals: _TraverseInternalsType = [('element', InternalTraversal.dp_clauseelement)]
    element: FromClause

    def __init__(self, element: FromClause):
        self.element = coercions.expect(roles.FromClauseRole, element)

    def _init_collections(self) -> None:
        pass

    @util.ro_non_memoized_property
    def columns(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        return self.element.columns

    @util.ro_non_memoized_property
    def c(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        return self.element.columns

    @property
    def primary_key(self) -> Iterable[NamedColumn[Any]]:
        return self.element.primary_key

    @property
    def foreign_keys(self) -> Iterable[ForeignKey]:
        return self.element.foreign_keys

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        return self.element.is_derived_from(fromclause)

    def alias(self, name: Optional[str]=None, flat: bool=False) -> NamedFromGrouping:
        return NamedFromGrouping(self.element.alias(name=name, flat=flat))

    def _anonymous_fromclause(self, **kw: Any) -> FromGrouping:
        return FromGrouping(self.element._anonymous_fromclause(**kw))

    @util.ro_non_memoized_property
    def _hide_froms(self) -> Iterable[FromClause]:
        return self.element._hide_froms

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return self.element._from_objects

    def __getstate__(self) -> Dict[str, FromClause]:
        return {'element': self.element}

    def __setstate__(self, state: Dict[str, FromClause]) -> None:
        self.element = state['element']