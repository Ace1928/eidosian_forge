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
class GenerativeSelect(SelectBase, Generative):
    """Base class for SELECT statements where additional elements can be
    added.

    This serves as the base for :class:`_expression.Select` and
    :class:`_expression.CompoundSelect`
    where elements such as ORDER BY, GROUP BY can be added and column
    rendering can be controlled.  Compare to
    :class:`_expression.TextualSelect`, which,
    while it subclasses :class:`_expression.SelectBase`
    and is also a SELECT construct,
    represents a fixed textual string which cannot be altered at this level,
    only wrapped as a subquery.

    """
    _order_by_clauses: Tuple[ColumnElement[Any], ...] = ()
    _group_by_clauses: Tuple[ColumnElement[Any], ...] = ()
    _limit_clause: Optional[ColumnElement[Any]] = None
    _offset_clause: Optional[ColumnElement[Any]] = None
    _fetch_clause: Optional[ColumnElement[Any]] = None
    _fetch_clause_options: Optional[Dict[str, bool]] = None
    _for_update_arg: Optional[ForUpdateArg] = None

    def __init__(self, _label_style: SelectLabelStyle=LABEL_STYLE_DEFAULT):
        self._label_style = _label_style

    @_generative
    def with_for_update(self, *, nowait: bool=False, read: bool=False, of: Optional[_ForUpdateOfArgument]=None, skip_locked: bool=False, key_share: bool=False) -> Self:
        """Specify a ``FOR UPDATE`` clause for this
        :class:`_expression.GenerativeSelect`.

        E.g.::

            stmt = select(table).with_for_update(nowait=True)

        On a database like PostgreSQL or Oracle, the above would render a
        statement like::

            SELECT table.a, table.b FROM table FOR UPDATE NOWAIT

        on other backends, the ``nowait`` option is ignored and instead
        would produce::

            SELECT table.a, table.b FROM table FOR UPDATE

        When called with no arguments, the statement will render with
        the suffix ``FOR UPDATE``.   Additional arguments can then be
        provided which allow for common database-specific
        variants.

        :param nowait: boolean; will render ``FOR UPDATE NOWAIT`` on Oracle
         and PostgreSQL dialects.

        :param read: boolean; will render ``LOCK IN SHARE MODE`` on MySQL,
         ``FOR SHARE`` on PostgreSQL.  On PostgreSQL, when combined with
         ``nowait``, will render ``FOR SHARE NOWAIT``.

        :param of: SQL expression or list of SQL expression elements,
         (typically :class:`_schema.Column` objects or a compatible expression,
         for some backends may also be a table expression) which will render
         into a ``FOR UPDATE OF`` clause; supported by PostgreSQL, Oracle, some
         MySQL versions and possibly others. May render as a table or as a
         column depending on backend.

        :param skip_locked: boolean, will render ``FOR UPDATE SKIP LOCKED``
         on Oracle and PostgreSQL dialects or ``FOR SHARE SKIP LOCKED`` if
         ``read=True`` is also specified.

        :param key_share: boolean, will render ``FOR NO KEY UPDATE``,
         or if combined with ``read=True`` will render ``FOR KEY SHARE``,
         on the PostgreSQL dialect.

        """
        self._for_update_arg = ForUpdateArg(nowait=nowait, read=read, of=of, skip_locked=skip_locked, key_share=key_share)
        return self

    def get_label_style(self) -> SelectLabelStyle:
        """
        Retrieve the current label style.

        .. versionadded:: 1.4

        """
        return self._label_style

    def set_label_style(self, style: SelectLabelStyle) -> Self:
        """Return a new selectable with the specified label style.

        There are three "label styles" available,
        :attr:`_sql.SelectLabelStyle.LABEL_STYLE_DISAMBIGUATE_ONLY`,
        :attr:`_sql.SelectLabelStyle.LABEL_STYLE_TABLENAME_PLUS_COL`, and
        :attr:`_sql.SelectLabelStyle.LABEL_STYLE_NONE`.   The default style is
        :attr:`_sql.SelectLabelStyle.LABEL_STYLE_TABLENAME_PLUS_COL`.

        In modern SQLAlchemy, there is not generally a need to change the
        labeling style, as per-expression labels are more effectively used by
        making use of the :meth:`_sql.ColumnElement.label` method. In past
        versions, :data:`_sql.LABEL_STYLE_TABLENAME_PLUS_COL` was used to
        disambiguate same-named columns from different tables, aliases, or
        subqueries; the newer :data:`_sql.LABEL_STYLE_DISAMBIGUATE_ONLY` now
        applies labels only to names that conflict with an existing name so
        that the impact of this labeling is minimal.

        The rationale for disambiguation is mostly so that all column
        expressions are available from a given :attr:`_sql.FromClause.c`
        collection when a subquery is created.

        .. versionadded:: 1.4 - the
            :meth:`_sql.GenerativeSelect.set_label_style` method replaces the
            previous combination of ``.apply_labels()``, ``.with_labels()`` and
            ``use_labels=True`` methods and/or parameters.

        .. seealso::

            :data:`_sql.LABEL_STYLE_DISAMBIGUATE_ONLY`

            :data:`_sql.LABEL_STYLE_TABLENAME_PLUS_COL`

            :data:`_sql.LABEL_STYLE_NONE`

            :data:`_sql.LABEL_STYLE_DEFAULT`

        """
        if self._label_style is not style:
            self = self._generate()
            self._label_style = style
        return self

    @property
    def _group_by_clause(self) -> ClauseList:
        """ClauseList access to group_by_clauses for legacy dialects"""
        return ClauseList._construct_raw(operators.comma_op, self._group_by_clauses)

    @property
    def _order_by_clause(self) -> ClauseList:
        """ClauseList access to order_by_clauses for legacy dialects"""
        return ClauseList._construct_raw(operators.comma_op, self._order_by_clauses)

    def _offset_or_limit_clause(self, element: _LimitOffsetType, name: Optional[str]=None, type_: Optional[_TypeEngineArgument[int]]=None) -> ColumnElement[Any]:
        """Convert the given value to an "offset or limit" clause.

        This handles incoming integers and converts to an expression; if
        an expression is already given, it is passed through.

        """
        return coercions.expect(roles.LimitOffsetRole, element, name=name, type_=type_)

    @overload
    def _offset_or_limit_clause_asint(self, clause: ColumnElement[Any], attrname: str) -> NoReturn:
        ...

    @overload
    def _offset_or_limit_clause_asint(self, clause: Optional[_OffsetLimitParam], attrname: str) -> Optional[int]:
        ...

    def _offset_or_limit_clause_asint(self, clause: Optional[ColumnElement[Any]], attrname: str) -> Union[NoReturn, Optional[int]]:
        """Convert the "offset or limit" clause of a select construct to an
        integer.

        This is only possible if the value is stored as a simple bound
        parameter. Otherwise, a compilation error is raised.

        """
        if clause is None:
            return None
        try:
            value = clause._limit_offset_value
        except AttributeError as err:
            raise exc.CompileError('This SELECT structure does not use a simple integer value for %s' % attrname) from err
        else:
            return util.asint(value)

    @property
    def _limit(self) -> Optional[int]:
        """Get an integer value for the limit.  This should only be used
        by code that cannot support a limit as a BindParameter or
        other custom clause as it will throw an exception if the limit
        isn't currently set to an integer.

        """
        return self._offset_or_limit_clause_asint(self._limit_clause, 'limit')

    def _simple_int_clause(self, clause: ClauseElement) -> bool:
        """True if the clause is a simple integer, False
        if it is not present or is a SQL expression.
        """
        return isinstance(clause, _OffsetLimitParam)

    @property
    def _offset(self) -> Optional[int]:
        """Get an integer value for the offset.  This should only be used
        by code that cannot support an offset as a BindParameter or
        other custom clause as it will throw an exception if the
        offset isn't currently set to an integer.

        """
        return self._offset_or_limit_clause_asint(self._offset_clause, 'offset')

    @property
    def _has_row_limiting_clause(self) -> bool:
        return self._limit_clause is not None or self._offset_clause is not None or self._fetch_clause is not None

    @_generative
    def limit(self, limit: _LimitOffsetType) -> Self:
        """Return a new selectable with the given LIMIT criterion
        applied.

        This is a numerical value which usually renders as a ``LIMIT``
        expression in the resulting select.  Backends that don't
        support ``LIMIT`` will attempt to provide similar
        functionality.

        .. note::

           The :meth:`_sql.GenerativeSelect.limit` method will replace
           any clause applied with :meth:`_sql.GenerativeSelect.fetch`.

        :param limit: an integer LIMIT parameter, or a SQL expression
         that provides an integer result. Pass ``None`` to reset it.

        .. seealso::

           :meth:`_sql.GenerativeSelect.fetch`

           :meth:`_sql.GenerativeSelect.offset`

        """
        self._fetch_clause = self._fetch_clause_options = None
        self._limit_clause = self._offset_or_limit_clause(limit)
        return self

    @_generative
    def fetch(self, count: _LimitOffsetType, with_ties: bool=False, percent: bool=False) -> Self:
        """Return a new selectable with the given FETCH FIRST criterion
        applied.

        This is a numeric value which usually renders as
        ``FETCH {FIRST | NEXT} [ count ] {ROW | ROWS} {ONLY | WITH TIES}``
        expression in the resulting select. This functionality is
        is currently implemented for Oracle, PostgreSQL, MSSQL.

        Use :meth:`_sql.GenerativeSelect.offset` to specify the offset.

        .. note::

           The :meth:`_sql.GenerativeSelect.fetch` method will replace
           any clause applied with :meth:`_sql.GenerativeSelect.limit`.

        .. versionadded:: 1.4

        :param count: an integer COUNT parameter, or a SQL expression
         that provides an integer result. When ``percent=True`` this will
         represent the percentage of rows to return, not the absolute value.
         Pass ``None`` to reset it.

        :param with_ties: When ``True``, the WITH TIES option is used
         to return any additional rows that tie for the last place in the
         result set according to the ``ORDER BY`` clause. The
         ``ORDER BY`` may be mandatory in this case. Defaults to ``False``

        :param percent: When ``True``, ``count`` represents the percentage
         of the total number of selected rows to return. Defaults to ``False``

        .. seealso::

           :meth:`_sql.GenerativeSelect.limit`

           :meth:`_sql.GenerativeSelect.offset`

        """
        self._limit_clause = None
        if count is None:
            self._fetch_clause = self._fetch_clause_options = None
        else:
            self._fetch_clause = self._offset_or_limit_clause(count)
            self._fetch_clause_options = {'with_ties': with_ties, 'percent': percent}
        return self

    @_generative
    def offset(self, offset: _LimitOffsetType) -> Self:
        """Return a new selectable with the given OFFSET criterion
        applied.


        This is a numeric value which usually renders as an ``OFFSET``
        expression in the resulting select.  Backends that don't
        support ``OFFSET`` will attempt to provide similar
        functionality.

        :param offset: an integer OFFSET parameter, or a SQL expression
         that provides an integer result. Pass ``None`` to reset it.

        .. seealso::

           :meth:`_sql.GenerativeSelect.limit`

           :meth:`_sql.GenerativeSelect.fetch`

        """
        self._offset_clause = self._offset_or_limit_clause(offset)
        return self

    @_generative
    @util.preload_module('sqlalchemy.sql.util')
    def slice(self, start: int, stop: int) -> Self:
        """Apply LIMIT / OFFSET to this statement based on a slice.

        The start and stop indices behave like the argument to Python's
        built-in :func:`range` function. This method provides an
        alternative to using ``LIMIT``/``OFFSET`` to get a slice of the
        query.

        For example, ::

            stmt = select(User).order_by(User).id.slice(1, 3)

        renders as

        .. sourcecode:: sql

           SELECT users.id AS users_id,
                  users.name AS users_name
           FROM users ORDER BY users.id
           LIMIT ? OFFSET ?
           (2, 1)

        .. note::

           The :meth:`_sql.GenerativeSelect.slice` method will replace
           any clause applied with :meth:`_sql.GenerativeSelect.fetch`.

        .. versionadded:: 1.4  Added the :meth:`_sql.GenerativeSelect.slice`
           method generalized from the ORM.

        .. seealso::

           :meth:`_sql.GenerativeSelect.limit`

           :meth:`_sql.GenerativeSelect.offset`

           :meth:`_sql.GenerativeSelect.fetch`

        """
        sql_util = util.preloaded.sql_util
        self._fetch_clause = self._fetch_clause_options = None
        self._limit_clause, self._offset_clause = sql_util._make_slice(self._limit_clause, self._offset_clause, start, stop)
        return self

    @_generative
    def order_by(self, __first: Union[Literal[None, _NoArg.NO_ARG], _ColumnExpressionOrStrLabelArgument[Any]]=_NoArg.NO_ARG, *clauses: _ColumnExpressionOrStrLabelArgument[Any]) -> Self:
        """Return a new selectable with the given list of ORDER BY
        criteria applied.

        e.g.::

            stmt = select(table).order_by(table.c.id, table.c.name)

        Calling this method multiple times is equivalent to calling it once
        with all the clauses concatenated. All existing ORDER BY criteria may
        be cancelled by passing ``None`` by itself.  New ORDER BY criteria may
        then be added by invoking :meth:`_orm.Query.order_by` again, e.g.::

            # will erase all ORDER BY and ORDER BY new_col alone
            stmt = stmt.order_by(None).order_by(new_col)

        :param \\*clauses: a series of :class:`_expression.ColumnElement`
         constructs
         which will be used to generate an ORDER BY clause.

        .. seealso::

            :ref:`tutorial_order_by` - in the :ref:`unified_tutorial`

            :ref:`tutorial_order_by_label` - in the :ref:`unified_tutorial`

        """
        if not clauses and __first is None:
            self._order_by_clauses = ()
        elif __first is not _NoArg.NO_ARG:
            self._order_by_clauses += tuple((coercions.expect(roles.OrderByRole, clause, apply_propagate_attrs=self) for clause in (__first,) + clauses))
        return self

    @_generative
    def group_by(self, __first: Union[Literal[None, _NoArg.NO_ARG], _ColumnExpressionOrStrLabelArgument[Any]]=_NoArg.NO_ARG, *clauses: _ColumnExpressionOrStrLabelArgument[Any]) -> Self:
        """Return a new selectable with the given list of GROUP BY
        criterion applied.

        All existing GROUP BY settings can be suppressed by passing ``None``.

        e.g.::

            stmt = select(table.c.name, func.max(table.c.stat)).\\
            group_by(table.c.name)

        :param \\*clauses: a series of :class:`_expression.ColumnElement`
         constructs
         which will be used to generate an GROUP BY clause.

        .. seealso::

            :ref:`tutorial_group_by_w_aggregates` - in the
            :ref:`unified_tutorial`

            :ref:`tutorial_order_by_label` - in the :ref:`unified_tutorial`

        """
        if not clauses and __first is None:
            self._group_by_clauses = ()
        elif __first is not _NoArg.NO_ARG:
            self._group_by_clauses += tuple((coercions.expect(roles.GroupByRole, clause, apply_propagate_attrs=self) for clause in (__first,) + clauses))
        return self