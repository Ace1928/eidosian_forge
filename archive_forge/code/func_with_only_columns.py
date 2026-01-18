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
@_generative
def with_only_columns(self, *entities: _ColumnsClauseArgument[Any], maintain_column_froms: bool=False, **__kw: Any) -> Select[Any]:
    """Return a new :func:`_expression.select` construct with its columns
        clause replaced with the given entities.

        By default, this method is exactly equivalent to as if the original
        :func:`_expression.select` had been called with the given entities.
        E.g. a statement::

            s = select(table1.c.a, table1.c.b)
            s = s.with_only_columns(table1.c.b)

        should be exactly equivalent to::

            s = select(table1.c.b)

        In this mode of operation, :meth:`_sql.Select.with_only_columns`
        will also dynamically alter the FROM clause of the
        statement if it is not explicitly stated.
        To maintain the existing set of FROMs including those implied by the
        current columns clause, add the
        :paramref:`_sql.Select.with_only_columns.maintain_column_froms`
        parameter::

            s = select(table1.c.a, table2.c.b)
            s = s.with_only_columns(table1.c.a, maintain_column_froms=True)

        The above parameter performs a transfer of the effective FROMs
        in the columns collection to the :meth:`_sql.Select.select_from`
        method, as though the following were invoked::

            s = select(table1.c.a, table2.c.b)
            s = s.select_from(table1, table2).with_only_columns(table1.c.a)

        The :paramref:`_sql.Select.with_only_columns.maintain_column_froms`
        parameter makes use of the :attr:`_sql.Select.columns_clause_froms`
        collection and performs an operation equivalent to the following::

            s = select(table1.c.a, table2.c.b)
            s = s.select_from(*s.columns_clause_froms).with_only_columns(table1.c.a)

        :param \\*entities: column expressions to be used.

        :param maintain_column_froms: boolean parameter that will ensure the
         FROM list implied from the current columns clause will be transferred
         to the :meth:`_sql.Select.select_from` method first.

         .. versionadded:: 1.4.23

        """
    if __kw:
        raise _no_kw()
    self._assert_no_memoizations()
    if maintain_column_froms:
        self.select_from.non_generative(self, *self.columns_clause_froms)
    _MemoizedSelectEntities._generate_for_statement(self)
    self._raw_columns = [coercions.expect(roles.ColumnsClauseRole, c) for c in coercions._expression_collection_was_a_list('entities', 'Select.with_only_columns', entities)]
    return self