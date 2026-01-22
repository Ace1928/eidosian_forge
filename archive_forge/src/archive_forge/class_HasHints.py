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
class HasHints:
    _hints: util.immutabledict[Tuple[FromClause, str], str] = util.immutabledict()
    _statement_hints: Tuple[Tuple[str, str], ...] = ()
    _has_hints_traverse_internals: _TraverseInternalsType = [('_statement_hints', InternalTraversal.dp_statement_hint_list), ('_hints', InternalTraversal.dp_table_hint_list)]

    def with_statement_hint(self, text: str, dialect_name: str='*') -> Self:
        """Add a statement hint to this :class:`_expression.Select` or
        other selectable object.

        This method is similar to :meth:`_expression.Select.with_hint`
        except that
        it does not require an individual table, and instead applies to the
        statement as a whole.

        Hints here are specific to the backend database and may include
        directives such as isolation levels, file directives, fetch directives,
        etc.

        .. seealso::

            :meth:`_expression.Select.with_hint`

            :meth:`_expression.Select.prefix_with` - generic SELECT prefixing
            which also can suit some database-specific HINT syntaxes such as
            MySQL optimizer hints

        """
        return self._with_hint(None, text, dialect_name)

    @_generative
    def with_hint(self, selectable: _FromClauseArgument, text: str, dialect_name: str='*') -> Self:
        """Add an indexing or other executional context hint for the given
        selectable to this :class:`_expression.Select` or other selectable
        object.

        The text of the hint is rendered in the appropriate
        location for the database backend in use, relative
        to the given :class:`_schema.Table` or :class:`_expression.Alias`
        passed as the
        ``selectable`` argument. The dialect implementation
        typically uses Python string substitution syntax
        with the token ``%(name)s`` to render the name of
        the table or alias. E.g. when using Oracle, the
        following::

            select(mytable).\\
                with_hint(mytable, "index(%(name)s ix_mytable)")

        Would render SQL as::

            select /*+ index(mytable ix_mytable) */ ... from mytable

        The ``dialect_name`` option will limit the rendering of a particular
        hint to a particular backend. Such as, to add hints for both Oracle
        and Sybase simultaneously::

            select(mytable).\\
                with_hint(mytable, "index(%(name)s ix_mytable)", 'oracle').\\
                with_hint(mytable, "WITH INDEX ix_mytable", 'mssql')

        .. seealso::

            :meth:`_expression.Select.with_statement_hint`

        """
        return self._with_hint(selectable, text, dialect_name)

    def _with_hint(self, selectable: Optional[_FromClauseArgument], text: str, dialect_name: str) -> Self:
        if selectable is None:
            self._statement_hints += ((dialect_name, text),)
        else:
            self._hints = self._hints.union({(coercions.expect(roles.FromClauseRole, selectable), dialect_name): text})
        return self