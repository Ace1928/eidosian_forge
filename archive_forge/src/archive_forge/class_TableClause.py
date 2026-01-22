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
class TableClause(roles.DMLTableRole, Immutable, NamedFromClause):
    """Represents a minimal "table" construct.

    This is a lightweight table object that has only a name, a
    collection of columns, which are typically produced
    by the :func:`_expression.column` function, and a schema::

        from sqlalchemy import table, column

        user = table("user",
                column("id"),
                column("name"),
                column("description"),
        )

    The :class:`_expression.TableClause` construct serves as the base for
    the more commonly used :class:`_schema.Table` object, providing
    the usual set of :class:`_expression.FromClause` services including
    the ``.c.`` collection and statement generation methods.

    It does **not** provide all the additional schema-level services
    of :class:`_schema.Table`, including constraints, references to other
    tables, or support for :class:`_schema.MetaData`-level services.
    It's useful
    on its own as an ad-hoc construct used to generate quick SQL
    statements when a more fully fledged :class:`_schema.Table`
    is not on hand.

    """
    __visit_name__ = 'table'
    _traverse_internals: _TraverseInternalsType = [('columns', InternalTraversal.dp_fromclause_canonical_column_collection), ('name', InternalTraversal.dp_string), ('schema', InternalTraversal.dp_string)]
    _is_table = True
    fullname: str
    implicit_returning = False
    ":class:`_expression.TableClause`\n    doesn't support having a primary key or column\n    -level defaults, so implicit returning doesn't apply."

    @util.ro_memoized_property
    def _autoincrement_column(self) -> Optional[ColumnClause[Any]]:
        """No PK or default support so no autoincrement column."""
        return None

    def __init__(self, name: str, *columns: ColumnClause[Any], **kw: Any):
        super().__init__()
        self.name = name
        self._columns = DedupeColumnCollection()
        self.primary_key = ColumnSet()
        self.foreign_keys = set()
        for c in columns:
            self.append_column(c)
        schema = kw.pop('schema', None)
        if schema is not None:
            self.schema = schema
        if self.schema is not None:
            self.fullname = '%s.%s' % (self.schema, self.name)
        else:
            self.fullname = self.name
        if kw:
            raise exc.ArgumentError('Unsupported argument(s): %s' % list(kw))
    if TYPE_CHECKING:

        @util.ro_non_memoized_property
        def columns(self) -> ReadOnlyColumnCollection[str, ColumnClause[Any]]:
            ...

        @util.ro_non_memoized_property
        def c(self) -> ReadOnlyColumnCollection[str, ColumnClause[Any]]:
            ...

    def __str__(self) -> str:
        if self.schema is not None:
            return self.schema + '.' + self.name
        else:
            return self.name

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        pass

    def _init_collections(self) -> None:
        pass

    @util.ro_memoized_property
    def description(self) -> str:
        return self.name

    def append_column(self, c: ColumnClause[Any]) -> None:
        existing = c.table
        if existing is not None and existing is not self:
            raise exc.ArgumentError("column object '%s' already assigned to table '%s'" % (c.key, existing))
        self._columns.add(c)
        c.table = self

    @util.preload_module('sqlalchemy.sql.dml')
    def insert(self) -> util.preloaded.sql_dml.Insert:
        """Generate an :class:`_sql.Insert` construct against this
        :class:`_expression.TableClause`.

        E.g.::

            table.insert().values(name='foo')

        See :func:`_expression.insert` for argument and usage information.

        """
        return util.preloaded.sql_dml.Insert(self)

    @util.preload_module('sqlalchemy.sql.dml')
    def update(self) -> Update:
        """Generate an :func:`_expression.update` construct against this
        :class:`_expression.TableClause`.

        E.g.::

            table.update().where(table.c.id==7).values(name='foo')

        See :func:`_expression.update` for argument and usage information.

        """
        return util.preloaded.sql_dml.Update(self)

    @util.preload_module('sqlalchemy.sql.dml')
    def delete(self) -> Delete:
        """Generate a :func:`_expression.delete` construct against this
        :class:`_expression.TableClause`.

        E.g.::

            table.delete().where(table.c.id==7)

        See :func:`_expression.delete` for argument and usage information.

        """
        return util.preloaded.sql_dml.Delete(self)

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return [self]