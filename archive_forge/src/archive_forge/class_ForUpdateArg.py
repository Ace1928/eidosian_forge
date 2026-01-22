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
class ForUpdateArg(ClauseElement):
    _traverse_internals: _TraverseInternalsType = [('of', InternalTraversal.dp_clauseelement_list), ('nowait', InternalTraversal.dp_boolean), ('read', InternalTraversal.dp_boolean), ('skip_locked', InternalTraversal.dp_boolean)]
    of: Optional[Sequence[ClauseElement]]
    nowait: bool
    read: bool
    skip_locked: bool

    @classmethod
    def _from_argument(cls, with_for_update: ForUpdateParameter) -> Optional[ForUpdateArg]:
        if isinstance(with_for_update, ForUpdateArg):
            return with_for_update
        elif with_for_update in (None, False):
            return None
        elif with_for_update is True:
            return ForUpdateArg()
        else:
            return ForUpdateArg(**cast('Dict[str, Any]', with_for_update))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ForUpdateArg) and other.nowait == self.nowait and (other.read == self.read) and (other.skip_locked == self.skip_locked) and (other.key_share == self.key_share) and (other.of is self.of)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return id(self)

    def __init__(self, *, nowait: bool=False, read: bool=False, of: Optional[_ForUpdateOfArgument]=None, skip_locked: bool=False, key_share: bool=False):
        """Represents arguments specified to
        :meth:`_expression.Select.for_update`.

        """
        self.nowait = nowait
        self.read = read
        self.skip_locked = skip_locked
        self.key_share = key_share
        if of is not None:
            self.of = [coercions.expect(roles.ColumnsClauseRole, elem) for elem in util.to_list(of)]
        else:
            self.of = None