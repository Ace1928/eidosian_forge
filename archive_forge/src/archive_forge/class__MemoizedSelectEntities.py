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
class _MemoizedSelectEntities(cache_key.HasCacheKey, traversals.HasCopyInternals, visitors.Traversible):
    """represents partial state from a Select object, for the case
    where Select.columns() has redefined the set of columns/entities the
    statement will be SELECTing from.  This object represents
    the entities from the SELECT before that transformation was applied,
    so that transformations that were made in terms of the SELECT at that
    time, such as join() as well as options(), can access the correct context.

    In previous SQLAlchemy versions, this wasn't needed because these
    constructs calculated everything up front, like when you called join()
    or options(), it did everything to figure out how that would translate
    into specific SQL constructs that would be ready to send directly to the
    SQL compiler when needed.  But as of
    1.4, all of that stuff is done in the compilation phase, during the
    "compile state" portion of the process, so that the work can all be
    cached.  So it needs to be able to resolve joins/options2 based on what
    the list of entities was when those methods were called.


    """
    __visit_name__ = 'memoized_select_entities'
    _traverse_internals: _TraverseInternalsType = [('_raw_columns', InternalTraversal.dp_clauseelement_list), ('_setup_joins', InternalTraversal.dp_setup_join_tuple), ('_with_options', InternalTraversal.dp_executable_options)]
    _is_clone_of: Optional[ClauseElement]
    _raw_columns: List[_ColumnsClauseElement]
    _setup_joins: Tuple[_SetupJoinsElement, ...]
    _with_options: Tuple[ExecutableOption, ...]
    _annotations = util.EMPTY_DICT

    def _clone(self, **kw: Any) -> Self:
        c = self.__class__.__new__(self.__class__)
        c.__dict__ = {k: v for k, v in self.__dict__.items()}
        c._is_clone_of = self.__dict__.get('_is_clone_of', self)
        return c

    @classmethod
    def _generate_for_statement(cls, select_stmt: Select[Any]) -> None:
        if select_stmt._setup_joins or select_stmt._with_options:
            self = _MemoizedSelectEntities()
            self._raw_columns = select_stmt._raw_columns
            self._setup_joins = select_stmt._setup_joins
            self._with_options = select_stmt._with_options
            select_stmt._memoized_select_entities += (self,)
            select_stmt._raw_columns = []
            select_stmt._setup_joins = select_stmt._with_options = ()