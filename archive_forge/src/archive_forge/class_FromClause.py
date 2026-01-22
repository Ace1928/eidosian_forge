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
class FromClause(roles.AnonymizedFromClauseRole, Selectable):
    """Represent an element that can be used within the ``FROM``
    clause of a ``SELECT`` statement.

    The most common forms of :class:`_expression.FromClause` are the
    :class:`_schema.Table` and the :func:`_expression.select` constructs.  Key
    features common to all :class:`_expression.FromClause` objects include:

    * a :attr:`.c` collection, which provides per-name access to a collection
      of :class:`_expression.ColumnElement` objects.
    * a :attr:`.primary_key` attribute, which is a collection of all those
      :class:`_expression.ColumnElement`
      objects that indicate the ``primary_key`` flag.
    * Methods to generate various derivations of a "from" clause, including
      :meth:`_expression.FromClause.alias`,
      :meth:`_expression.FromClause.join`,
      :meth:`_expression.FromClause.select`.


    """
    __visit_name__ = 'fromclause'
    named_with_column = False

    @util.ro_non_memoized_property
    def _hide_froms(self) -> Iterable[FromClause]:
        return ()
    _is_clone_of: Optional[FromClause]
    _columns: ColumnCollection[Any, Any]
    schema: Optional[str] = None
    "Define the 'schema' attribute for this :class:`_expression.FromClause`.\n\n    This is typically ``None`` for most objects except that of\n    :class:`_schema.Table`, where it is taken as the value of the\n    :paramref:`_schema.Table.schema` argument.\n\n    "
    is_selectable = True
    _is_from_clause = True
    _is_join = False
    _use_schema_map = False

    def select(self) -> Select[Any]:
        """Return a SELECT of this :class:`_expression.FromClause`.


        e.g.::

            stmt = some_table.select().where(some_table.c.id == 5)

        .. seealso::

            :func:`_expression.select` - general purpose
            method which allows for arbitrary column lists.

        """
        return Select(self)

    def join(self, right: _FromClauseArgument, onclause: Optional[_ColumnExpressionArgument[bool]]=None, isouter: bool=False, full: bool=False) -> Join:
        """Return a :class:`_expression.Join` from this
        :class:`_expression.FromClause`
        to another :class:`FromClause`.

        E.g.::

            from sqlalchemy import join

            j = user_table.join(address_table,
                            user_table.c.id == address_table.c.user_id)
            stmt = select(user_table).select_from(j)

        would emit SQL along the lines of::

            SELECT user.id, user.name FROM user
            JOIN address ON user.id = address.user_id

        :param right: the right side of the join; this is any
         :class:`_expression.FromClause` object such as a
         :class:`_schema.Table` object, and
         may also be a selectable-compatible object such as an ORM-mapped
         class.

        :param onclause: a SQL expression representing the ON clause of the
         join.  If left at ``None``, :meth:`_expression.FromClause.join`
         will attempt to
         join the two tables based on a foreign key relationship.

        :param isouter: if True, render a LEFT OUTER JOIN, instead of JOIN.

        :param full: if True, render a FULL OUTER JOIN, instead of LEFT OUTER
         JOIN.  Implies :paramref:`.FromClause.join.isouter`.

        .. seealso::

            :func:`_expression.join` - standalone function

            :class:`_expression.Join` - the type of object produced

        """
        return Join(self, right, onclause, isouter, full)

    def outerjoin(self, right: _FromClauseArgument, onclause: Optional[_ColumnExpressionArgument[bool]]=None, full: bool=False) -> Join:
        """Return a :class:`_expression.Join` from this
        :class:`_expression.FromClause`
        to another :class:`FromClause`, with the "isouter" flag set to
        True.

        E.g.::

            from sqlalchemy import outerjoin

            j = user_table.outerjoin(address_table,
                            user_table.c.id == address_table.c.user_id)

        The above is equivalent to::

            j = user_table.join(
                address_table,
                user_table.c.id == address_table.c.user_id,
                isouter=True)

        :param right: the right side of the join; this is any
         :class:`_expression.FromClause` object such as a
         :class:`_schema.Table` object, and
         may also be a selectable-compatible object such as an ORM-mapped
         class.

        :param onclause: a SQL expression representing the ON clause of the
         join.  If left at ``None``, :meth:`_expression.FromClause.join`
         will attempt to
         join the two tables based on a foreign key relationship.

        :param full: if True, render a FULL OUTER JOIN, instead of
         LEFT OUTER JOIN.

        .. seealso::

            :meth:`_expression.FromClause.join`

            :class:`_expression.Join`

        """
        return Join(self, right, onclause, True, full)

    def alias(self, name: Optional[str]=None, flat: bool=False) -> NamedFromClause:
        """Return an alias of this :class:`_expression.FromClause`.

        E.g.::

            a2 = some_table.alias('a2')

        The above code creates an :class:`_expression.Alias`
        object which can be used
        as a FROM clause in any SELECT statement.

        .. seealso::

            :ref:`tutorial_using_aliases`

            :func:`_expression.alias`

        """
        return Alias._construct(self, name=name)

    def tablesample(self, sampling: Union[float, Function[Any]], name: Optional[str]=None, seed: Optional[roles.ExpressionElementRole[Any]]=None) -> TableSample:
        """Return a TABLESAMPLE alias of this :class:`_expression.FromClause`.

        The return value is the :class:`_expression.TableSample`
        construct also
        provided by the top-level :func:`_expression.tablesample` function.

        .. seealso::

            :func:`_expression.tablesample` - usage guidelines and parameters

        """
        return TableSample._construct(self, sampling=sampling, name=name, seed=seed)

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        """Return ``True`` if this :class:`_expression.FromClause` is
        'derived' from the given ``FromClause``.

        An example would be an Alias of a Table is derived from that Table.

        """
        return fromclause in self._cloned_set

    def _is_lexical_equivalent(self, other: FromClause) -> bool:
        """Return ``True`` if this :class:`_expression.FromClause` and
        the other represent the same lexical identity.

        This tests if either one is a copy of the other, or
        if they are the same via annotation identity.

        """
        return bool(self._cloned_set.intersection(other._cloned_set))

    @util.ro_non_memoized_property
    def description(self) -> str:
        """A brief description of this :class:`_expression.FromClause`.

        Used primarily for error message formatting.

        """
        return getattr(self, 'name', self.__class__.__name__ + ' object')

    def _generate_fromclause_column_proxies(self, fromclause: FromClause) -> None:
        fromclause._columns._populate_separate_keys((col._make_proxy(fromclause) for col in self.c))

    @util.ro_non_memoized_property
    def exported_columns(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        """A :class:`_expression.ColumnCollection`
        that represents the "exported"
        columns of this :class:`_expression.Selectable`.

        The "exported" columns for a :class:`_expression.FromClause`
        object are synonymous
        with the :attr:`_expression.FromClause.columns` collection.

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_expression.Selectable.exported_columns`

            :attr:`_expression.SelectBase.exported_columns`


        """
        return self.c

    @util.ro_non_memoized_property
    def columns(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        """A named-based collection of :class:`_expression.ColumnElement`
        objects maintained by this :class:`_expression.FromClause`.

        The :attr:`.columns`, or :attr:`.c` collection, is the gateway
        to the construction of SQL expressions using table-bound or
        other selectable-bound columns::

            select(mytable).where(mytable.c.somecolumn == 5)

        :return: a :class:`.ColumnCollection` object.

        """
        return self.c

    @util.ro_memoized_property
    def c(self) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        """
        A synonym for :attr:`.FromClause.columns`

        :return: a :class:`.ColumnCollection`

        """
        if '_columns' not in self.__dict__:
            self._init_collections()
            self._populate_column_collection()
        return self._columns.as_readonly()

    @util.ro_non_memoized_property
    def entity_namespace(self) -> _EntityNamespace:
        """Return a namespace used for name-based access in SQL expressions.

        This is the namespace that is used to resolve "filter_by()" type
        expressions, such as::

            stmt.filter_by(address='some address')

        It defaults to the ``.c`` collection, however internally it can
        be overridden using the "entity_namespace" annotation to deliver
        alternative results.

        """
        return self.c

    @util.ro_memoized_property
    def primary_key(self) -> Iterable[NamedColumn[Any]]:
        """Return the iterable collection of :class:`_schema.Column` objects
        which comprise the primary key of this :class:`_selectable.FromClause`.

        For a :class:`_schema.Table` object, this collection is represented
        by the :class:`_schema.PrimaryKeyConstraint` which itself is an
        iterable collection of :class:`_schema.Column` objects.

        """
        self._init_collections()
        self._populate_column_collection()
        return self.primary_key

    @util.ro_memoized_property
    def foreign_keys(self) -> Iterable[ForeignKey]:
        """Return the collection of :class:`_schema.ForeignKey` marker objects
        which this FromClause references.

        Each :class:`_schema.ForeignKey` is a member of a
        :class:`_schema.Table`-wide
        :class:`_schema.ForeignKeyConstraint`.

        .. seealso::

            :attr:`_schema.Table.foreign_key_constraints`

        """
        self._init_collections()
        self._populate_column_collection()
        return self.foreign_keys

    def _reset_column_collection(self) -> None:
        """Reset the attributes linked to the ``FromClause.c`` attribute.

        This collection is separate from all the other memoized things
        as it has shown to be sensitive to being cleared out in situations
        where enclosing code, typically in a replacement traversal scenario,
        has already established strong relationships
        with the exported columns.

        The collection is cleared for the case where a table is having a
        column added to it as well as within a Join during copy internals.

        """
        for key in ['_columns', 'columns', 'c', 'primary_key', 'foreign_keys']:
            self.__dict__.pop(key, None)

    @util.ro_non_memoized_property
    def _select_iterable(self) -> _SelectIterable:
        return (c for c in self.c if not _never_select_column(c))

    def _init_collections(self) -> None:
        assert '_columns' not in self.__dict__
        assert 'primary_key' not in self.__dict__
        assert 'foreign_keys' not in self.__dict__
        self._columns = ColumnCollection()
        self.primary_key = ColumnSet()
        self.foreign_keys = set()

    @property
    def _cols_populated(self) -> bool:
        return '_columns' in self.__dict__

    def _populate_column_collection(self) -> None:
        """Called on subclasses to establish the .c collection.

        Each implementation has a different way of establishing
        this collection.

        """

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        """Given a column added to the .c collection of an underlying
        selectable, produce the local version of that column, assuming this
        selectable ultimately should proxy this column.

        this is used to "ping" a derived selectable to add a new column
        to its .c. collection when a Column has been added to one of the
        Table objects it ultimately derives from.

        If the given selectable hasn't populated its .c. collection yet,
        it should at least pass on the message to the contained selectables,
        but it will return None.

        This method is currently used by Declarative to allow Table
        columns to be added to a partially constructed inheritance
        mapping that may have already produced joins.  The method
        isn't public right now, as the full span of implications
        and/or caveats aren't yet clear.

        It's also possible that this functionality could be invoked by
        default via an event, which would require that
        selectables maintain a weak referencing collection of all
        derivations.

        """
        self._reset_column_collection()

    def _anonymous_fromclause(self, *, name: Optional[str]=None, flat: bool=False) -> FromClause:
        return self.alias(name=name)
    if TYPE_CHECKING:

        def self_group(self, against: Optional[OperatorType]=None) -> Union[FromGrouping, Self]:
            ...