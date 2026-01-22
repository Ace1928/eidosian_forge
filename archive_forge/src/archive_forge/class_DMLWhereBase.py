from __future__ import annotations
import collections.abc as collections_abc
import operator
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import roles
from . import util as sql_util
from ._typing import _TP
from ._typing import _unexpected_kw
from ._typing import is_column_element
from ._typing import is_named_from_clause
from .base import _entity_namespace_key
from .base import _exclusive_against
from .base import _from_objects
from .base import _generative
from .base import _select_iterables
from .base import ColumnCollection
from .base import CompileState
from .base import DialectKWArgs
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Null
from .selectable import Alias
from .selectable import ExecutableReturnsRows
from .selectable import FromClause
from .selectable import HasCTE
from .selectable import HasPrefixes
from .selectable import Join
from .selectable import SelectLabelStyle
from .selectable import TableClause
from .selectable import TypedReturnsRows
from .sqltypes import NullType
from .visitors import InternalTraversal
from .. import exc
from .. import util
from ..util.typing import Self
from ..util.typing import TypeGuard
class DMLWhereBase:
    table: _DMLTableElement
    _where_criteria: Tuple[ColumnElement[Any], ...] = ()

    @_generative
    def where(self, *whereclause: _ColumnExpressionArgument[bool]) -> Self:
        """Return a new construct with the given expression(s) added to
        its WHERE clause, joined to the existing clause via AND, if any.

        Both :meth:`_dml.Update.where` and :meth:`_dml.Delete.where`
        support multiple-table forms, including database-specific
        ``UPDATE...FROM`` as well as ``DELETE..USING``.  For backends that
        don't have multiple-table support, a backend agnostic approach
        to using multiple tables is to make use of correlated subqueries.
        See the linked tutorial sections below for examples.

        .. seealso::

            :ref:`tutorial_correlated_updates`

            :ref:`tutorial_update_from`

            :ref:`tutorial_multi_table_deletes`

        """
        for criterion in whereclause:
            where_criteria: ColumnElement[Any] = coercions.expect(roles.WhereHavingRole, criterion, apply_propagate_attrs=self)
            self._where_criteria += (where_criteria,)
        return self

    def filter(self, *criteria: roles.ExpressionElementRole[Any]) -> Self:
        """A synonym for the :meth:`_dml.DMLWhereBase.where` method.

        .. versionadded:: 1.4

        """
        return self.where(*criteria)

    def _filter_by_zero(self) -> _DMLTableElement:
        return self.table

    def filter_by(self, **kwargs: Any) -> Self:
        """apply the given filtering criterion as a WHERE clause
        to this select.

        """
        from_entity = self._filter_by_zero()
        clauses = [_entity_namespace_key(from_entity, key) == value for key, value in kwargs.items()]
        return self.filter(*clauses)

    @property
    def whereclause(self) -> Optional[ColumnElement[Any]]:
        """Return the completed WHERE clause for this :class:`.DMLWhereBase`
        statement.

        This assembles the current collection of WHERE criteria
        into a single :class:`_expression.BooleanClauseList` construct.


        .. versionadded:: 1.4

        """
        return BooleanClauseList._construct_for_whereclause(self._where_criteria)