from __future__ import annotations
from enum import IntEnum
from operator import add as _uncast_add
from operator import and_ as _uncast_and_
from operator import contains as _uncast_contains
from operator import eq as _uncast_eq
from operator import floordiv as _uncast_floordiv
from operator import ge as _uncast_ge
from operator import getitem as _uncast_getitem
from operator import gt as _uncast_gt
from operator import inv as _uncast_inv
from operator import le as _uncast_le
from operator import lshift as _uncast_lshift
from operator import lt as _uncast_lt
from operator import mod as _uncast_mod
from operator import mul as _uncast_mul
from operator import ne as _uncast_ne
from operator import neg as _uncast_neg
from operator import or_ as _uncast_or_
from operator import rshift as _uncast_rshift
from operator import sub as _uncast_sub
from operator import truediv as _uncast_truediv
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import exc
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
class ColumnOperators(Operators):
    """Defines boolean, comparison, and other operators for
    :class:`_expression.ColumnElement` expressions.

    By default, all methods call down to
    :meth:`.operate` or :meth:`.reverse_operate`,
    passing in the appropriate operator function from the
    Python builtin ``operator`` module or
    a SQLAlchemy-specific operator function from
    :mod:`sqlalchemy.expression.operators`.   For example
    the ``__eq__`` function::

        def __eq__(self, other):
            return self.operate(operators.eq, other)

    Where ``operators.eq`` is essentially::

        def eq(a, b):
            return a == b

    The core column expression unit :class:`_expression.ColumnElement`
    overrides :meth:`.Operators.operate` and others
    to return further :class:`_expression.ColumnElement` constructs,
    so that the ``==`` operation above is replaced by a clause
    construct.

    .. seealso::

        :ref:`types_operators`

        :attr:`.TypeEngine.comparator_factory`

        :class:`.ColumnOperators`

        :class:`.PropComparator`

    """
    __slots__ = ()
    timetuple: Literal[None] = None
    'Hack, allows datetime objects to be compared on the LHS.'
    if typing.TYPE_CHECKING:

        def operate(self, op: OperatorType, *other: Any, **kwargs: Any) -> ColumnOperators:
            ...

        def reverse_operate(self, op: OperatorType, other: Any, **kwargs: Any) -> ColumnOperators:
            ...

    def __lt__(self, other: Any) -> ColumnOperators:
        """Implement the ``<`` operator.

        In a column context, produces the clause ``a < b``.

        """
        return self.operate(lt, other)

    def __le__(self, other: Any) -> ColumnOperators:
        """Implement the ``<=`` operator.

        In a column context, produces the clause ``a <= b``.

        """
        return self.operate(le, other)
    if TYPE_CHECKING:

        def __hash__(self) -> int:
            ...
    else:
        __hash__ = Operators.__hash__

    def __eq__(self, other: Any) -> ColumnOperators:
        """Implement the ``==`` operator.

        In a column context, produces the clause ``a = b``.
        If the target is ``None``, produces ``a IS NULL``.

        """
        return self.operate(eq, other)

    def __ne__(self, other: Any) -> ColumnOperators:
        """Implement the ``!=`` operator.

        In a column context, produces the clause ``a != b``.
        If the target is ``None``, produces ``a IS NOT NULL``.

        """
        return self.operate(ne, other)

    def is_distinct_from(self, other: Any) -> ColumnOperators:
        """Implement the ``IS DISTINCT FROM`` operator.

        Renders "a IS DISTINCT FROM b" on most platforms;
        on some such as SQLite may render "a IS NOT b".

        """
        return self.operate(is_distinct_from, other)

    def is_not_distinct_from(self, other: Any) -> ColumnOperators:
        """Implement the ``IS NOT DISTINCT FROM`` operator.

        Renders "a IS NOT DISTINCT FROM b" on most platforms;
        on some such as SQLite may render "a IS b".

        .. versionchanged:: 1.4 The ``is_not_distinct_from()`` operator is
           renamed from ``isnot_distinct_from()`` in previous releases.
           The previous name remains available for backwards compatibility.

        """
        return self.operate(is_not_distinct_from, other)
    if TYPE_CHECKING:

        def isnot_distinct_from(self, other: Any) -> ColumnOperators:
            ...
    else:
        isnot_distinct_from = is_not_distinct_from

    def __gt__(self, other: Any) -> ColumnOperators:
        """Implement the ``>`` operator.

        In a column context, produces the clause ``a > b``.

        """
        return self.operate(gt, other)

    def __ge__(self, other: Any) -> ColumnOperators:
        """Implement the ``>=`` operator.

        In a column context, produces the clause ``a >= b``.

        """
        return self.operate(ge, other)

    def __neg__(self) -> ColumnOperators:
        """Implement the ``-`` operator.

        In a column context, produces the clause ``-a``.

        """
        return self.operate(neg)

    def __contains__(self, other: Any) -> ColumnOperators:
        return self.operate(contains, other)

    def __getitem__(self, index: Any) -> ColumnOperators:
        """Implement the [] operator.

        This can be used by some database-specific types
        such as PostgreSQL ARRAY and HSTORE.

        """
        return self.operate(getitem, index)

    def __lshift__(self, other: Any) -> ColumnOperators:
        """implement the << operator.

        Not used by SQLAlchemy core, this is provided
        for custom operator systems which want to use
        << as an extension point.
        """
        return self.operate(lshift, other)

    def __rshift__(self, other: Any) -> ColumnOperators:
        """implement the >> operator.

        Not used by SQLAlchemy core, this is provided
        for custom operator systems which want to use
        >> as an extension point.
        """
        return self.operate(rshift, other)

    def concat(self, other: Any) -> ColumnOperators:
        """Implement the 'concat' operator.

        In a column context, produces the clause ``a || b``,
        or uses the ``concat()`` operator on MySQL.

        """
        return self.operate(concat_op, other)

    def _rconcat(self, other: Any) -> ColumnOperators:
        """Implement an 'rconcat' operator.

        this is for internal use at the moment

        .. versionadded:: 1.4.40

        """
        return self.reverse_operate(concat_op, other)

    def like(self, other: Any, escape: Optional[str]=None) -> ColumnOperators:
        """Implement the ``like`` operator.

        In a column context, produces the expression::

            a LIKE other

        E.g.::

            stmt = select(sometable).\\
                where(sometable.c.column.like("%foobar%"))

        :param other: expression to be compared
        :param escape: optional escape character, renders the ``ESCAPE``
          keyword, e.g.::

            somecolumn.like("foo/%bar", escape="/")

        .. seealso::

            :meth:`.ColumnOperators.ilike`

        """
        return self.operate(like_op, other, escape=escape)

    def ilike(self, other: Any, escape: Optional[str]=None) -> ColumnOperators:
        """Implement the ``ilike`` operator, e.g. case insensitive LIKE.

        In a column context, produces an expression either of the form::

            lower(a) LIKE lower(other)

        Or on backends that support the ILIKE operator::

            a ILIKE other

        E.g.::

            stmt = select(sometable).\\
                where(sometable.c.column.ilike("%foobar%"))

        :param other: expression to be compared
        :param escape: optional escape character, renders the ``ESCAPE``
          keyword, e.g.::

            somecolumn.ilike("foo/%bar", escape="/")

        .. seealso::

            :meth:`.ColumnOperators.like`

        """
        return self.operate(ilike_op, other, escape=escape)

    def bitwise_xor(self, other: Any) -> ColumnOperators:
        """Produce a bitwise XOR operation, typically via the ``^``
        operator, or ``#`` for PostgreSQL.

        .. versionadded:: 2.0.2

        .. seealso::

            :ref:`operators_bitwise`

        """
        return self.operate(bitwise_xor_op, other)

    def bitwise_or(self, other: Any) -> ColumnOperators:
        """Produce a bitwise OR operation, typically via the ``|``
        operator.

        .. versionadded:: 2.0.2

        .. seealso::

            :ref:`operators_bitwise`

        """
        return self.operate(bitwise_or_op, other)

    def bitwise_and(self, other: Any) -> ColumnOperators:
        """Produce a bitwise AND operation, typically via the ``&``
        operator.

        .. versionadded:: 2.0.2

        .. seealso::

            :ref:`operators_bitwise`

        """
        return self.operate(bitwise_and_op, other)

    def bitwise_not(self) -> ColumnOperators:
        """Produce a bitwise NOT operation, typically via the ``~``
        operator.

        .. versionadded:: 2.0.2

        .. seealso::

            :ref:`operators_bitwise`

        """
        return self.operate(bitwise_not_op)

    def bitwise_lshift(self, other: Any) -> ColumnOperators:
        """Produce a bitwise LSHIFT operation, typically via the ``<<``
        operator.

        .. versionadded:: 2.0.2

        .. seealso::

            :ref:`operators_bitwise`

        """
        return self.operate(bitwise_lshift_op, other)

    def bitwise_rshift(self, other: Any) -> ColumnOperators:
        """Produce a bitwise RSHIFT operation, typically via the ``>>``
        operator.

        .. versionadded:: 2.0.2

        .. seealso::

            :ref:`operators_bitwise`

        """
        return self.operate(bitwise_rshift_op, other)

    def in_(self, other: Any) -> ColumnOperators:
        """Implement the ``in`` operator.

        In a column context, produces the clause ``column IN <other>``.

        The given parameter ``other`` may be:

        * A list of literal values, e.g.::

            stmt.where(column.in_([1, 2, 3]))

          In this calling form, the list of items is converted to a set of
          bound parameters the same length as the list given::

            WHERE COL IN (?, ?, ?)

        * A list of tuples may be provided if the comparison is against a
          :func:`.tuple_` containing multiple expressions::

            from sqlalchemy import tuple_
            stmt.where(tuple_(col1, col2).in_([(1, 10), (2, 20), (3, 30)]))

        * An empty list, e.g.::

            stmt.where(column.in_([]))

          In this calling form, the expression renders an "empty set"
          expression.  These expressions are tailored to individual backends
          and are generally trying to get an empty SELECT statement as a
          subquery.  Such as on SQLite, the expression is::

            WHERE col IN (SELECT 1 FROM (SELECT 1) WHERE 1!=1)

          .. versionchanged:: 1.4  empty IN expressions now use an
             execution-time generated SELECT subquery in all cases.

        * A bound parameter, e.g. :func:`.bindparam`, may be used if it
          includes the :paramref:`.bindparam.expanding` flag::

            stmt.where(column.in_(bindparam('value', expanding=True)))

          In this calling form, the expression renders a special non-SQL
          placeholder expression that looks like::

            WHERE COL IN ([EXPANDING_value])

          This placeholder expression is intercepted at statement execution
          time to be converted into the variable number of bound parameter
          form illustrated earlier.   If the statement were executed as::

            connection.execute(stmt, {"value": [1, 2, 3]})

          The database would be passed a bound parameter for each value::

            WHERE COL IN (?, ?, ?)

          .. versionadded:: 1.2 added "expanding" bound parameters

          If an empty list is passed, a special "empty list" expression,
          which is specific to the database in use, is rendered.  On
          SQLite this would be::

            WHERE COL IN (SELECT 1 FROM (SELECT 1) WHERE 1!=1)

          .. versionadded:: 1.3 "expanding" bound parameters now support
             empty lists

        * a :func:`_expression.select` construct, which is usually a
          correlated scalar select::

            stmt.where(
                column.in_(
                    select(othertable.c.y).
                    where(table.c.x == othertable.c.x)
                )
            )

          In this calling form, :meth:`.ColumnOperators.in_` renders as given::

            WHERE COL IN (SELECT othertable.y
            FROM othertable WHERE othertable.x = table.x)

        :param other: a list of literals, a :func:`_expression.select`
         construct, or a :func:`.bindparam` construct that includes the
         :paramref:`.bindparam.expanding` flag set to True.

        """
        return self.operate(in_op, other)

    def not_in(self, other: Any) -> ColumnOperators:
        """implement the ``NOT IN`` operator.

        This is equivalent to using negation with
        :meth:`.ColumnOperators.in_`, i.e. ``~x.in_(y)``.

        In the case that ``other`` is an empty sequence, the compiler
        produces an "empty not in" expression.   This defaults to the
        expression "1 = 1" to produce true in all cases.  The
        :paramref:`_sa.create_engine.empty_in_strategy` may be used to
        alter this behavior.

        .. versionchanged:: 1.4 The ``not_in()`` operator is renamed from
           ``notin_()`` in previous releases.  The previous name remains
           available for backwards compatibility.

        .. versionchanged:: 1.2  The :meth:`.ColumnOperators.in_` and
           :meth:`.ColumnOperators.not_in` operators
           now produce a "static" expression for an empty IN sequence
           by default.

        .. seealso::

            :meth:`.ColumnOperators.in_`

        """
        return self.operate(not_in_op, other)
    if TYPE_CHECKING:

        def notin_(self, other: Any) -> ColumnOperators:
            ...
    else:
        notin_ = not_in

    def not_like(self, other: Any, escape: Optional[str]=None) -> ColumnOperators:
        """implement the ``NOT LIKE`` operator.

        This is equivalent to using negation with
        :meth:`.ColumnOperators.like`, i.e. ``~x.like(y)``.

        .. versionchanged:: 1.4 The ``not_like()`` operator is renamed from
           ``notlike()`` in previous releases.  The previous name remains
           available for backwards compatibility.

        .. seealso::

            :meth:`.ColumnOperators.like`

        """
        return self.operate(not_like_op, other, escape=escape)
    if TYPE_CHECKING:

        def notlike(self, other: Any, escape: Optional[str]=None) -> ColumnOperators:
            ...
    else:
        notlike = not_like

    def not_ilike(self, other: Any, escape: Optional[str]=None) -> ColumnOperators:
        """implement the ``NOT ILIKE`` operator.

        This is equivalent to using negation with
        :meth:`.ColumnOperators.ilike`, i.e. ``~x.ilike(y)``.

        .. versionchanged:: 1.4 The ``not_ilike()`` operator is renamed from
           ``notilike()`` in previous releases.  The previous name remains
           available for backwards compatibility.

        .. seealso::

            :meth:`.ColumnOperators.ilike`

        """
        return self.operate(not_ilike_op, other, escape=escape)
    if TYPE_CHECKING:

        def notilike(self, other: Any, escape: Optional[str]=None) -> ColumnOperators:
            ...
    else:
        notilike = not_ilike

    def is_(self, other: Any) -> ColumnOperators:
        """Implement the ``IS`` operator.

        Normally, ``IS`` is generated automatically when comparing to a
        value of ``None``, which resolves to ``NULL``.  However, explicit
        usage of ``IS`` may be desirable if comparing to boolean values
        on certain platforms.

        .. seealso:: :meth:`.ColumnOperators.is_not`

        """
        return self.operate(is_, other)

    def is_not(self, other: Any) -> ColumnOperators:
        """Implement the ``IS NOT`` operator.

        Normally, ``IS NOT`` is generated automatically when comparing to a
        value of ``None``, which resolves to ``NULL``.  However, explicit
        usage of ``IS NOT`` may be desirable if comparing to boolean values
        on certain platforms.

        .. versionchanged:: 1.4 The ``is_not()`` operator is renamed from
           ``isnot()`` in previous releases.  The previous name remains
           available for backwards compatibility.

        .. seealso:: :meth:`.ColumnOperators.is_`

        """
        return self.operate(is_not, other)
    if TYPE_CHECKING:

        def isnot(self, other: Any) -> ColumnOperators:
            ...
    else:
        isnot = is_not

    def startswith(self, other: Any, escape: Optional[str]=None, autoescape: bool=False) -> ColumnOperators:
        """Implement the ``startswith`` operator.

        Produces a LIKE expression that tests against a match for the start
        of a string value::

            column LIKE <other> || '%'

        E.g.::

            stmt = select(sometable).\\
                where(sometable.c.column.startswith("foobar"))

        Since the operator uses ``LIKE``, wildcard characters
        ``"%"`` and ``"_"`` that are present inside the <other> expression
        will behave like wildcards as well.   For literal string
        values, the :paramref:`.ColumnOperators.startswith.autoescape` flag
        may be set to ``True`` to apply escaping to occurrences of these
        characters within the string value so that they match as themselves
        and not as wildcard characters.  Alternatively, the
        :paramref:`.ColumnOperators.startswith.escape` parameter will establish
        a given character as an escape character which can be of use when
        the target expression is not a literal string.

        :param other: expression to be compared.   This is usually a plain
          string value, but can also be an arbitrary SQL expression.  LIKE
          wildcard characters ``%`` and ``_`` are not escaped by default unless
          the :paramref:`.ColumnOperators.startswith.autoescape` flag is
          set to True.

        :param autoescape: boolean; when True, establishes an escape character
          within the LIKE expression, then applies it to all occurrences of
          ``"%"``, ``"_"`` and the escape character itself within the
          comparison value, which is assumed to be a literal string and not a
          SQL expression.

          An expression such as::

            somecolumn.startswith("foo%bar", autoescape=True)

          Will render as::

            somecolumn LIKE :param || '%' ESCAPE '/'

          With the value of ``:param`` as ``"foo/%bar"``.

        :param escape: a character which when given will render with the
          ``ESCAPE`` keyword to establish that character as the escape
          character.  This character can then be placed preceding occurrences
          of ``%`` and ``_`` to allow them to act as themselves and not
          wildcard characters.

          An expression such as::

            somecolumn.startswith("foo/%bar", escape="^")

          Will render as::

            somecolumn LIKE :param || '%' ESCAPE '^'

          The parameter may also be combined with
          :paramref:`.ColumnOperators.startswith.autoescape`::

            somecolumn.startswith("foo%bar^bat", escape="^", autoescape=True)

          Where above, the given literal parameter will be converted to
          ``"foo^%bar^^bat"`` before being passed to the database.

        .. seealso::

            :meth:`.ColumnOperators.endswith`

            :meth:`.ColumnOperators.contains`

            :meth:`.ColumnOperators.like`

        """
        return self.operate(startswith_op, other, escape=escape, autoescape=autoescape)

    def istartswith(self, other: Any, escape: Optional[str]=None, autoescape: bool=False) -> ColumnOperators:
        """Implement the ``istartswith`` operator, e.g. case insensitive
        version of :meth:`.ColumnOperators.startswith`.

        Produces a LIKE expression that tests against an insensitive
        match for the start of a string value::

            lower(column) LIKE lower(<other>) || '%'

        E.g.::

            stmt = select(sometable).\\
                where(sometable.c.column.istartswith("foobar"))

        Since the operator uses ``LIKE``, wildcard characters
        ``"%"`` and ``"_"`` that are present inside the <other> expression
        will behave like wildcards as well.   For literal string
        values, the :paramref:`.ColumnOperators.istartswith.autoescape` flag
        may be set to ``True`` to apply escaping to occurrences of these
        characters within the string value so that they match as themselves
        and not as wildcard characters.  Alternatively, the
        :paramref:`.ColumnOperators.istartswith.escape` parameter will
        establish a given character as an escape character which can be of
        use when the target expression is not a literal string.

        :param other: expression to be compared.   This is usually a plain
          string value, but can also be an arbitrary SQL expression.  LIKE
          wildcard characters ``%`` and ``_`` are not escaped by default unless
          the :paramref:`.ColumnOperators.istartswith.autoescape` flag is
          set to True.

        :param autoescape: boolean; when True, establishes an escape character
          within the LIKE expression, then applies it to all occurrences of
          ``"%"``, ``"_"`` and the escape character itself within the
          comparison value, which is assumed to be a literal string and not a
          SQL expression.

          An expression such as::

            somecolumn.istartswith("foo%bar", autoescape=True)

          Will render as::

            lower(somecolumn) LIKE lower(:param) || '%' ESCAPE '/'

          With the value of ``:param`` as ``"foo/%bar"``.

        :param escape: a character which when given will render with the
          ``ESCAPE`` keyword to establish that character as the escape
          character.  This character can then be placed preceding occurrences
          of ``%`` and ``_`` to allow them to act as themselves and not
          wildcard characters.

          An expression such as::

            somecolumn.istartswith("foo/%bar", escape="^")

          Will render as::

            lower(somecolumn) LIKE lower(:param) || '%' ESCAPE '^'

          The parameter may also be combined with
          :paramref:`.ColumnOperators.istartswith.autoescape`::

            somecolumn.istartswith("foo%bar^bat", escape="^", autoescape=True)

          Where above, the given literal parameter will be converted to
          ``"foo^%bar^^bat"`` before being passed to the database.

        .. seealso::

            :meth:`.ColumnOperators.startswith`
        """
        return self.operate(istartswith_op, other, escape=escape, autoescape=autoescape)

    def endswith(self, other: Any, escape: Optional[str]=None, autoescape: bool=False) -> ColumnOperators:
        """Implement the 'endswith' operator.

        Produces a LIKE expression that tests against a match for the end
        of a string value::

            column LIKE '%' || <other>

        E.g.::

            stmt = select(sometable).\\
                where(sometable.c.column.endswith("foobar"))

        Since the operator uses ``LIKE``, wildcard characters
        ``"%"`` and ``"_"`` that are present inside the <other> expression
        will behave like wildcards as well.   For literal string
        values, the :paramref:`.ColumnOperators.endswith.autoescape` flag
        may be set to ``True`` to apply escaping to occurrences of these
        characters within the string value so that they match as themselves
        and not as wildcard characters.  Alternatively, the
        :paramref:`.ColumnOperators.endswith.escape` parameter will establish
        a given character as an escape character which can be of use when
        the target expression is not a literal string.

        :param other: expression to be compared.   This is usually a plain
          string value, but can also be an arbitrary SQL expression.  LIKE
          wildcard characters ``%`` and ``_`` are not escaped by default unless
          the :paramref:`.ColumnOperators.endswith.autoescape` flag is
          set to True.

        :param autoescape: boolean; when True, establishes an escape character
          within the LIKE expression, then applies it to all occurrences of
          ``"%"``, ``"_"`` and the escape character itself within the
          comparison value, which is assumed to be a literal string and not a
          SQL expression.

          An expression such as::

            somecolumn.endswith("foo%bar", autoescape=True)

          Will render as::

            somecolumn LIKE '%' || :param ESCAPE '/'

          With the value of ``:param`` as ``"foo/%bar"``.

        :param escape: a character which when given will render with the
          ``ESCAPE`` keyword to establish that character as the escape
          character.  This character can then be placed preceding occurrences
          of ``%`` and ``_`` to allow them to act as themselves and not
          wildcard characters.

          An expression such as::

            somecolumn.endswith("foo/%bar", escape="^")

          Will render as::

            somecolumn LIKE '%' || :param ESCAPE '^'

          The parameter may also be combined with
          :paramref:`.ColumnOperators.endswith.autoescape`::

            somecolumn.endswith("foo%bar^bat", escape="^", autoescape=True)

          Where above, the given literal parameter will be converted to
          ``"foo^%bar^^bat"`` before being passed to the database.

        .. seealso::

            :meth:`.ColumnOperators.startswith`

            :meth:`.ColumnOperators.contains`

            :meth:`.ColumnOperators.like`

        """
        return self.operate(endswith_op, other, escape=escape, autoescape=autoescape)

    def iendswith(self, other: Any, escape: Optional[str]=None, autoescape: bool=False) -> ColumnOperators:
        """Implement the ``iendswith`` operator, e.g. case insensitive
        version of :meth:`.ColumnOperators.endswith`.

        Produces a LIKE expression that tests against an insensitive match
        for the end of a string value::

            lower(column) LIKE '%' || lower(<other>)

        E.g.::

            stmt = select(sometable).\\
                where(sometable.c.column.iendswith("foobar"))

        Since the operator uses ``LIKE``, wildcard characters
        ``"%"`` and ``"_"`` that are present inside the <other> expression
        will behave like wildcards as well.   For literal string
        values, the :paramref:`.ColumnOperators.iendswith.autoescape` flag
        may be set to ``True`` to apply escaping to occurrences of these
        characters within the string value so that they match as themselves
        and not as wildcard characters.  Alternatively, the
        :paramref:`.ColumnOperators.iendswith.escape` parameter will establish
        a given character as an escape character which can be of use when
        the target expression is not a literal string.

        :param other: expression to be compared.   This is usually a plain
          string value, but can also be an arbitrary SQL expression.  LIKE
          wildcard characters ``%`` and ``_`` are not escaped by default unless
          the :paramref:`.ColumnOperators.iendswith.autoescape` flag is
          set to True.

        :param autoescape: boolean; when True, establishes an escape character
          within the LIKE expression, then applies it to all occurrences of
          ``"%"``, ``"_"`` and the escape character itself within the
          comparison value, which is assumed to be a literal string and not a
          SQL expression.

          An expression such as::

            somecolumn.iendswith("foo%bar", autoescape=True)

          Will render as::

            lower(somecolumn) LIKE '%' || lower(:param) ESCAPE '/'

          With the value of ``:param`` as ``"foo/%bar"``.

        :param escape: a character which when given will render with the
          ``ESCAPE`` keyword to establish that character as the escape
          character.  This character can then be placed preceding occurrences
          of ``%`` and ``_`` to allow them to act as themselves and not
          wildcard characters.

          An expression such as::

            somecolumn.iendswith("foo/%bar", escape="^")

          Will render as::

            lower(somecolumn) LIKE '%' || lower(:param) ESCAPE '^'

          The parameter may also be combined with
          :paramref:`.ColumnOperators.iendswith.autoescape`::

            somecolumn.endswith("foo%bar^bat", escape="^", autoescape=True)

          Where above, the given literal parameter will be converted to
          ``"foo^%bar^^bat"`` before being passed to the database.

        .. seealso::

            :meth:`.ColumnOperators.endswith`
        """
        return self.operate(iendswith_op, other, escape=escape, autoescape=autoescape)

    def contains(self, other: Any, **kw: Any) -> ColumnOperators:
        """Implement the 'contains' operator.

        Produces a LIKE expression that tests against a match for the middle
        of a string value::

            column LIKE '%' || <other> || '%'

        E.g.::

            stmt = select(sometable).\\
                where(sometable.c.column.contains("foobar"))

        Since the operator uses ``LIKE``, wildcard characters
        ``"%"`` and ``"_"`` that are present inside the <other> expression
        will behave like wildcards as well.   For literal string
        values, the :paramref:`.ColumnOperators.contains.autoescape` flag
        may be set to ``True`` to apply escaping to occurrences of these
        characters within the string value so that they match as themselves
        and not as wildcard characters.  Alternatively, the
        :paramref:`.ColumnOperators.contains.escape` parameter will establish
        a given character as an escape character which can be of use when
        the target expression is not a literal string.

        :param other: expression to be compared.   This is usually a plain
          string value, but can also be an arbitrary SQL expression.  LIKE
          wildcard characters ``%`` and ``_`` are not escaped by default unless
          the :paramref:`.ColumnOperators.contains.autoescape` flag is
          set to True.

        :param autoescape: boolean; when True, establishes an escape character
          within the LIKE expression, then applies it to all occurrences of
          ``"%"``, ``"_"`` and the escape character itself within the
          comparison value, which is assumed to be a literal string and not a
          SQL expression.

          An expression such as::

            somecolumn.contains("foo%bar", autoescape=True)

          Will render as::

            somecolumn LIKE '%' || :param || '%' ESCAPE '/'

          With the value of ``:param`` as ``"foo/%bar"``.

        :param escape: a character which when given will render with the
          ``ESCAPE`` keyword to establish that character as the escape
          character.  This character can then be placed preceding occurrences
          of ``%`` and ``_`` to allow them to act as themselves and not
          wildcard characters.

          An expression such as::

            somecolumn.contains("foo/%bar", escape="^")

          Will render as::

            somecolumn LIKE '%' || :param || '%' ESCAPE '^'

          The parameter may also be combined with
          :paramref:`.ColumnOperators.contains.autoescape`::

            somecolumn.contains("foo%bar^bat", escape="^", autoescape=True)

          Where above, the given literal parameter will be converted to
          ``"foo^%bar^^bat"`` before being passed to the database.

        .. seealso::

            :meth:`.ColumnOperators.startswith`

            :meth:`.ColumnOperators.endswith`

            :meth:`.ColumnOperators.like`


        """
        return self.operate(contains_op, other, **kw)

    def icontains(self, other: Any, **kw: Any) -> ColumnOperators:
        """Implement the ``icontains`` operator, e.g. case insensitive
        version of :meth:`.ColumnOperators.contains`.

        Produces a LIKE expression that tests against an insensitive match
        for the middle of a string value::

            lower(column) LIKE '%' || lower(<other>) || '%'

        E.g.::

            stmt = select(sometable).\\
                where(sometable.c.column.icontains("foobar"))

        Since the operator uses ``LIKE``, wildcard characters
        ``"%"`` and ``"_"`` that are present inside the <other> expression
        will behave like wildcards as well.   For literal string
        values, the :paramref:`.ColumnOperators.icontains.autoescape` flag
        may be set to ``True`` to apply escaping to occurrences of these
        characters within the string value so that they match as themselves
        and not as wildcard characters.  Alternatively, the
        :paramref:`.ColumnOperators.icontains.escape` parameter will establish
        a given character as an escape character which can be of use when
        the target expression is not a literal string.

        :param other: expression to be compared.   This is usually a plain
          string value, but can also be an arbitrary SQL expression.  LIKE
          wildcard characters ``%`` and ``_`` are not escaped by default unless
          the :paramref:`.ColumnOperators.icontains.autoescape` flag is
          set to True.

        :param autoescape: boolean; when True, establishes an escape character
          within the LIKE expression, then applies it to all occurrences of
          ``"%"``, ``"_"`` and the escape character itself within the
          comparison value, which is assumed to be a literal string and not a
          SQL expression.

          An expression such as::

            somecolumn.icontains("foo%bar", autoescape=True)

          Will render as::

            lower(somecolumn) LIKE '%' || lower(:param) || '%' ESCAPE '/'

          With the value of ``:param`` as ``"foo/%bar"``.

        :param escape: a character which when given will render with the
          ``ESCAPE`` keyword to establish that character as the escape
          character.  This character can then be placed preceding occurrences
          of ``%`` and ``_`` to allow them to act as themselves and not
          wildcard characters.

          An expression such as::

            somecolumn.icontains("foo/%bar", escape="^")

          Will render as::

            lower(somecolumn) LIKE '%' || lower(:param) || '%' ESCAPE '^'

          The parameter may also be combined with
          :paramref:`.ColumnOperators.contains.autoescape`::

            somecolumn.icontains("foo%bar^bat", escape="^", autoescape=True)

          Where above, the given literal parameter will be converted to
          ``"foo^%bar^^bat"`` before being passed to the database.

        .. seealso::

            :meth:`.ColumnOperators.contains`

        """
        return self.operate(icontains_op, other, **kw)

    def match(self, other: Any, **kwargs: Any) -> ColumnOperators:
        """Implements a database-specific 'match' operator.

        :meth:`_sql.ColumnOperators.match` attempts to resolve to
        a MATCH-like function or operator provided by the backend.
        Examples include:

        * PostgreSQL - renders ``x @@ plainto_tsquery(y)``

            .. versionchanged:: 2.0  ``plainto_tsquery()`` is used instead
               of ``to_tsquery()`` for PostgreSQL now; for compatibility with
               other forms, see :ref:`postgresql_match`.


        * MySQL - renders ``MATCH (x) AGAINST (y IN BOOLEAN MODE)``

          .. seealso::

                :class:`_mysql.match` - MySQL specific construct with
                additional features.

        * Oracle - renders ``CONTAINS(x, y)``
        * other backends may provide special implementations.
        * Backends without any special implementation will emit
          the operator as "MATCH".  This is compatible with SQLite, for
          example.

        """
        return self.operate(match_op, other, **kwargs)

    def regexp_match(self, pattern: Any, flags: Optional[str]=None) -> ColumnOperators:
        """Implements a database-specific 'regexp match' operator.

        E.g.::

            stmt = select(table.c.some_column).where(
                table.c.some_column.regexp_match('^(b|c)')
            )

        :meth:`_sql.ColumnOperators.regexp_match` attempts to resolve to
        a REGEXP-like function or operator provided by the backend, however
        the specific regular expression syntax and flags available are
        **not backend agnostic**.

        Examples include:

        * PostgreSQL - renders ``x ~ y`` or ``x !~ y`` when negated.
        * Oracle - renders ``REGEXP_LIKE(x, y)``
        * SQLite - uses SQLite's ``REGEXP`` placeholder operator and calls into
          the Python ``re.match()`` builtin.
        * other backends may provide special implementations.
        * Backends without any special implementation will emit
          the operator as "REGEXP" or "NOT REGEXP".  This is compatible with
          SQLite and MySQL, for example.

        Regular expression support is currently implemented for Oracle,
        PostgreSQL, MySQL and MariaDB.  Partial support is available for
        SQLite.  Support among third-party dialects may vary.

        :param pattern: The regular expression pattern string or column
          clause.
        :param flags: Any regular expression string flags to apply, passed as
          plain Python string only.  These flags are backend specific.
          Some backends, like PostgreSQL and MariaDB, may alternatively
          specify the flags as part of the pattern.
          When using the ignore case flag 'i' in PostgreSQL, the ignore case
          regexp match operator ``~*`` or ``!~*`` will be used.

        .. versionadded:: 1.4

        .. versionchanged:: 1.4.48, 2.0.18  Note that due to an implementation
           error, the "flags" parameter previously accepted SQL expression
           objects such as column expressions in addition to plain Python
           strings.   This implementation did not work correctly with caching
           and was removed; strings only should be passed for the "flags"
           parameter, as these flags are rendered as literal inline values
           within SQL expressions.

        .. seealso::

            :meth:`_sql.ColumnOperators.regexp_replace`


        """
        return self.operate(regexp_match_op, pattern, flags=flags)

    def regexp_replace(self, pattern: Any, replacement: Any, flags: Optional[str]=None) -> ColumnOperators:
        """Implements a database-specific 'regexp replace' operator.

        E.g.::

            stmt = select(
                table.c.some_column.regexp_replace(
                    'b(..)',
                    'X\x01Y',
                    flags='g'
                )
            )

        :meth:`_sql.ColumnOperators.regexp_replace` attempts to resolve to
        a REGEXP_REPLACE-like function provided by the backend, that
        usually emit the function ``REGEXP_REPLACE()``.  However,
        the specific regular expression syntax and flags available are
        **not backend agnostic**.

        Regular expression replacement support is currently implemented for
        Oracle, PostgreSQL, MySQL 8 or greater and MariaDB.  Support among
        third-party dialects may vary.

        :param pattern: The regular expression pattern string or column
          clause.
        :param pattern: The replacement string or column clause.
        :param flags: Any regular expression string flags to apply, passed as
          plain Python string only.  These flags are backend specific.
          Some backends, like PostgreSQL and MariaDB, may alternatively
          specify the flags as part of the pattern.

        .. versionadded:: 1.4

        .. versionchanged:: 1.4.48, 2.0.18  Note that due to an implementation
           error, the "flags" parameter previously accepted SQL expression
           objects such as column expressions in addition to plain Python
           strings.   This implementation did not work correctly with caching
           and was removed; strings only should be passed for the "flags"
           parameter, as these flags are rendered as literal inline values
           within SQL expressions.


        .. seealso::

            :meth:`_sql.ColumnOperators.regexp_match`

        """
        return self.operate(regexp_replace_op, pattern, replacement=replacement, flags=flags)

    def desc(self) -> ColumnOperators:
        """Produce a :func:`_expression.desc` clause against the
        parent object."""
        return self.operate(desc_op)

    def asc(self) -> ColumnOperators:
        """Produce a :func:`_expression.asc` clause against the
        parent object."""
        return self.operate(asc_op)

    def nulls_first(self) -> ColumnOperators:
        """Produce a :func:`_expression.nulls_first` clause against the
        parent object.

        .. versionchanged:: 1.4 The ``nulls_first()`` operator is
           renamed from ``nullsfirst()`` in previous releases.
           The previous name remains available for backwards compatibility.
        """
        return self.operate(nulls_first_op)
    if TYPE_CHECKING:

        def nullsfirst(self) -> ColumnOperators:
            ...
    else:
        nullsfirst = nulls_first

    def nulls_last(self) -> ColumnOperators:
        """Produce a :func:`_expression.nulls_last` clause against the
        parent object.

        .. versionchanged:: 1.4 The ``nulls_last()`` operator is
           renamed from ``nullslast()`` in previous releases.
           The previous name remains available for backwards compatibility.
        """
        return self.operate(nulls_last_op)
    if TYPE_CHECKING:

        def nullslast(self) -> ColumnOperators:
            ...
    else:
        nullslast = nulls_last

    def collate(self, collation: str) -> ColumnOperators:
        """Produce a :func:`_expression.collate` clause against
        the parent object, given the collation string.

        .. seealso::

            :func:`_expression.collate`

        """
        return self.operate(collate, collation)

    def __radd__(self, other: Any) -> ColumnOperators:
        """Implement the ``+`` operator in reverse.

        See :meth:`.ColumnOperators.__add__`.

        """
        return self.reverse_operate(add, other)

    def __rsub__(self, other: Any) -> ColumnOperators:
        """Implement the ``-`` operator in reverse.

        See :meth:`.ColumnOperators.__sub__`.

        """
        return self.reverse_operate(sub, other)

    def __rmul__(self, other: Any) -> ColumnOperators:
        """Implement the ``*`` operator in reverse.

        See :meth:`.ColumnOperators.__mul__`.

        """
        return self.reverse_operate(mul, other)

    def __rmod__(self, other: Any) -> ColumnOperators:
        """Implement the ``%`` operator in reverse.

        See :meth:`.ColumnOperators.__mod__`.

        """
        return self.reverse_operate(mod, other)

    def between(self, cleft: Any, cright: Any, symmetric: bool=False) -> ColumnOperators:
        """Produce a :func:`_expression.between` clause against
        the parent object, given the lower and upper range.

        """
        return self.operate(between_op, cleft, cright, symmetric=symmetric)

    def distinct(self) -> ColumnOperators:
        """Produce a :func:`_expression.distinct` clause against the
        parent object.

        """
        return self.operate(distinct_op)

    def any_(self) -> ColumnOperators:
        """Produce an :func:`_expression.any_` clause against the
        parent object.

        See the documentation for :func:`_sql.any_` for examples.

        .. note:: be sure to not confuse the newer
            :meth:`_sql.ColumnOperators.any_` method with the **legacy**
            version of this method, the :meth:`_types.ARRAY.Comparator.any`
            method that's specific to :class:`_types.ARRAY`, which uses a
            different calling style.

        """
        return self.operate(any_op)

    def all_(self) -> ColumnOperators:
        """Produce an :func:`_expression.all_` clause against the
        parent object.

        See the documentation for :func:`_sql.all_` for examples.

        .. note:: be sure to not confuse the newer
            :meth:`_sql.ColumnOperators.all_` method with the **legacy**
            version of this method, the :meth:`_types.ARRAY.Comparator.all`
            method that's specific to :class:`_types.ARRAY`, which uses a
            different calling style.

        """
        return self.operate(all_op)

    def __add__(self, other: Any) -> ColumnOperators:
        """Implement the ``+`` operator.

        In a column context, produces the clause ``a + b``
        if the parent object has non-string affinity.
        If the parent object has a string affinity,
        produces the concatenation operator, ``a || b`` -
        see :meth:`.ColumnOperators.concat`.

        """
        return self.operate(add, other)

    def __sub__(self, other: Any) -> ColumnOperators:
        """Implement the ``-`` operator.

        In a column context, produces the clause ``a - b``.

        """
        return self.operate(sub, other)

    def __mul__(self, other: Any) -> ColumnOperators:
        """Implement the ``*`` operator.

        In a column context, produces the clause ``a * b``.

        """
        return self.operate(mul, other)

    def __mod__(self, other: Any) -> ColumnOperators:
        """Implement the ``%`` operator.

        In a column context, produces the clause ``a % b``.

        """
        return self.operate(mod, other)

    def __truediv__(self, other: Any) -> ColumnOperators:
        """Implement the ``/`` operator.

        In a column context, produces the clause ``a / b``, and
        considers the result type to be numeric.

        .. versionchanged:: 2.0  The truediv operator against two integers
           is now considered to return a numeric value.    Behavior on specific
           backends may vary.

        """
        return self.operate(truediv, other)

    def __rtruediv__(self, other: Any) -> ColumnOperators:
        """Implement the ``/`` operator in reverse.

        See :meth:`.ColumnOperators.__truediv__`.

        """
        return self.reverse_operate(truediv, other)

    def __floordiv__(self, other: Any) -> ColumnOperators:
        """Implement the ``//`` operator.

        In a column context, produces the clause ``a / b``,
        which is the same as "truediv", but considers the result
        type to be integer.

        .. versionadded:: 2.0

        """
        return self.operate(floordiv, other)

    def __rfloordiv__(self, other: Any) -> ColumnOperators:
        """Implement the ``//`` operator in reverse.

        See :meth:`.ColumnOperators.__floordiv__`.

        """
        return self.reverse_operate(floordiv, other)