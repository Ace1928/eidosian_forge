from __future__ import annotations
from decimal import Decimal
from enum import IntEnum
import itertools
import operator
import re
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple as typing_Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from ._typing import has_schema_attr
from ._typing import is_named_from_clause
from ._typing import is_quoted_name
from ._typing import is_tuple_type
from .annotation import Annotated
from .annotation import SupportsWrappingAnnotations
from .base import _clone
from .base import _expand_cloned
from .base import _generative
from .base import _NoArg
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .base import Immutable
from .base import NO_ARG
from .base import SingletonConstant
from .cache_key import MemoizedHasCacheKey
from .cache_key import NO_CACHE
from .coercions import _document_text_coercion  # noqa
from .operators import ColumnOperators
from .traversals import HasCopyInternals
from .visitors import cloned_traverse
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .visitors import traverse
from .visitors import Visitable
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util import TypingOnly
from ..util.typing import Literal
from ..util.typing import Self
class TextClause(roles.DDLConstraintColumnRole, roles.DDLExpressionRole, roles.StatementOptionRole, roles.WhereHavingRole, roles.OrderByRole, roles.FromClauseRole, roles.SelectStatementRole, roles.InElementRole, Generative, Executable, DQLDMLClauseElement, roles.BinaryElementRole[Any], inspection.Inspectable['TextClause']):
    """Represent a literal SQL text fragment.

    E.g.::

        from sqlalchemy import text

        t = text("SELECT * FROM users")
        result = connection.execute(t)


    The :class:`_expression.TextClause` construct is produced using the
    :func:`_expression.text`
    function; see that function for full documentation.

    .. seealso::

        :func:`_expression.text`

    """
    __visit_name__ = 'textclause'
    _traverse_internals: _TraverseInternalsType = [('_bindparams', InternalTraversal.dp_string_clauseelement_dict), ('text', InternalTraversal.dp_string)]
    _is_text_clause = True
    _is_textual = True
    _bind_params_regex = re.compile('(?<![:\\w\\x5c]):(\\w+)(?!:)', re.UNICODE)
    _is_implicitly_boolean = False
    _render_label_in_columns_clause = False
    _omit_from_statements = False
    _is_collection_aggregate = False

    @property
    def _hide_froms(self) -> Iterable[FromClause]:
        return ()

    def __and__(self, other):
        return and_(self, other)

    @property
    def _select_iterable(self) -> _SelectIterable:
        return (self,)
    key: Optional[str] = None
    _label: Optional[str] = None
    _allow_label_resolve = False

    @property
    def _is_star(self):
        return self.text == '*'

    def __init__(self, text: str):
        self._bindparams: Dict[str, BindParameter[Any]] = {}

        def repl(m):
            self._bindparams[m.group(1)] = BindParameter(m.group(1))
            return ':%s' % m.group(1)
        self.text = self._bind_params_regex.sub(repl, text)

    @_generative
    def bindparams(self, *binds: BindParameter[Any], **names_to_values: Any) -> Self:
        """Establish the values and/or types of bound parameters within
        this :class:`_expression.TextClause` construct.

        Given a text construct such as::

            from sqlalchemy import text
            stmt = text("SELECT id, name FROM user WHERE name=:name "
                        "AND timestamp=:timestamp")

        the :meth:`_expression.TextClause.bindparams`
        method can be used to establish
        the initial value of ``:name`` and ``:timestamp``,
        using simple keyword arguments::

            stmt = stmt.bindparams(name='jack',
                        timestamp=datetime.datetime(2012, 10, 8, 15, 12, 5))

        Where above, new :class:`.BindParameter` objects
        will be generated with the names ``name`` and ``timestamp``, and
        values of ``jack`` and ``datetime.datetime(2012, 10, 8, 15, 12, 5)``,
        respectively.  The types will be
        inferred from the values given, in this case :class:`.String` and
        :class:`.DateTime`.

        When specific typing behavior is needed, the positional ``*binds``
        argument can be used in which to specify :func:`.bindparam` constructs
        directly.  These constructs must include at least the ``key``
        argument, then an optional value and type::

            from sqlalchemy import bindparam
            stmt = stmt.bindparams(
                            bindparam('name', value='jack', type_=String),
                            bindparam('timestamp', type_=DateTime)
                        )

        Above, we specified the type of :class:`.DateTime` for the
        ``timestamp`` bind, and the type of :class:`.String` for the ``name``
        bind.  In the case of ``name`` we also set the default value of
        ``"jack"``.

        Additional bound parameters can be supplied at statement execution
        time, e.g.::

            result = connection.execute(stmt,
                        timestamp=datetime.datetime(2012, 10, 8, 15, 12, 5))

        The :meth:`_expression.TextClause.bindparams`
        method can be called repeatedly,
        where it will re-use existing :class:`.BindParameter` objects to add
        new information.  For example, we can call
        :meth:`_expression.TextClause.bindparams`
        first with typing information, and a
        second time with value information, and it will be combined::

            stmt = text("SELECT id, name FROM user WHERE name=:name "
                        "AND timestamp=:timestamp")
            stmt = stmt.bindparams(
                bindparam('name', type_=String),
                bindparam('timestamp', type_=DateTime)
            )
            stmt = stmt.bindparams(
                name='jack',
                timestamp=datetime.datetime(2012, 10, 8, 15, 12, 5)
            )

        The :meth:`_expression.TextClause.bindparams`
        method also supports the concept of
        **unique** bound parameters.  These are parameters that are
        "uniquified" on name at statement compilation time, so that  multiple
        :func:`_expression.text`
        constructs may be combined together without the names
        conflicting.  To use this feature, specify the
        :paramref:`.BindParameter.unique` flag on each :func:`.bindparam`
        object::

            stmt1 = text("select id from table where name=:name").bindparams(
                bindparam("name", value='name1', unique=True)
            )
            stmt2 = text("select id from table where name=:name").bindparams(
                bindparam("name", value='name2', unique=True)
            )

            union = union_all(
                stmt1.columns(column("id")),
                stmt2.columns(column("id"))
            )

        The above statement will render as::

            select id from table where name=:name_1
            UNION ALL select id from table where name=:name_2

        .. versionadded:: 1.3.11  Added support for the
           :paramref:`.BindParameter.unique` flag to work with
           :func:`_expression.text`
           constructs.

        """
        self._bindparams = new_params = self._bindparams.copy()
        for bind in binds:
            try:
                existing = new_params[bind._orig_key]
            except KeyError as err:
                raise exc.ArgumentError("This text() construct doesn't define a bound parameter named %r" % bind._orig_key) from err
            else:
                new_params[existing._orig_key] = bind
        for key, value in names_to_values.items():
            try:
                existing = new_params[key]
            except KeyError as err:
                raise exc.ArgumentError("This text() construct doesn't define a bound parameter named %r" % key) from err
            else:
                new_params[key] = existing._with_value(value, required=False)
        return self

    @util.preload_module('sqlalchemy.sql.selectable')
    def columns(self, *cols: _ColumnExpressionArgument[Any], **types: TypeEngine[Any]) -> TextualSelect:
        """Turn this :class:`_expression.TextClause` object into a
        :class:`_expression.TextualSelect`
        object that serves the same role as a SELECT
        statement.

        The :class:`_expression.TextualSelect` is part of the
        :class:`_expression.SelectBase`
        hierarchy and can be embedded into another statement by using the
        :meth:`_expression.TextualSelect.subquery` method to produce a
        :class:`.Subquery`
        object, which can then be SELECTed from.

        This function essentially bridges the gap between an entirely
        textual SELECT statement and the SQL expression language concept
        of a "selectable"::

            from sqlalchemy.sql import column, text

            stmt = text("SELECT id, name FROM some_table")
            stmt = stmt.columns(column('id'), column('name')).subquery('st')

            stmt = select(mytable).\\
                    select_from(
                        mytable.join(stmt, mytable.c.name == stmt.c.name)
                    ).where(stmt.c.id > 5)

        Above, we pass a series of :func:`_expression.column` elements to the
        :meth:`_expression.TextClause.columns` method positionally.  These
        :func:`_expression.column`
        elements now become first class elements upon the
        :attr:`_expression.TextualSelect.selected_columns` column collection,
        which then
        become part of the :attr:`.Subquery.c` collection after
        :meth:`_expression.TextualSelect.subquery` is invoked.

        The column expressions we pass to
        :meth:`_expression.TextClause.columns` may
        also be typed; when we do so, these :class:`.TypeEngine` objects become
        the effective return type of the column, so that SQLAlchemy's
        result-set-processing systems may be used on the return values.
        This is often needed for types such as date or boolean types, as well
        as for unicode processing on some dialect configurations::

            stmt = text("SELECT id, name, timestamp FROM some_table")
            stmt = stmt.columns(
                        column('id', Integer),
                        column('name', Unicode),
                        column('timestamp', DateTime)
                    )

            for id, name, timestamp in connection.execute(stmt):
                print(id, name, timestamp)

        As a shortcut to the above syntax, keyword arguments referring to
        types alone may be used, if only type conversion is needed::

            stmt = text("SELECT id, name, timestamp FROM some_table")
            stmt = stmt.columns(
                        id=Integer,
                        name=Unicode,
                        timestamp=DateTime
                    )

            for id, name, timestamp in connection.execute(stmt):
                print(id, name, timestamp)

        The positional form of :meth:`_expression.TextClause.columns`
        also provides the
        unique feature of **positional column targeting**, which is
        particularly useful when using the ORM with complex textual queries. If
        we specify the columns from our model to
        :meth:`_expression.TextClause.columns`,
        the result set will match to those columns positionally, meaning the
        name or origin of the column in the textual SQL doesn't matter::

            stmt = text("SELECT users.id, addresses.id, users.id, "
                 "users.name, addresses.email_address AS email "
                 "FROM users JOIN addresses ON users.id=addresses.user_id "
                 "WHERE users.id = 1").columns(
                    User.id,
                    Address.id,
                    Address.user_id,
                    User.name,
                    Address.email_address
                 )

            query = session.query(User).from_statement(stmt).options(
                contains_eager(User.addresses))

        The :meth:`_expression.TextClause.columns` method provides a direct
        route to calling :meth:`_expression.FromClause.subquery` as well as
        :meth:`_expression.SelectBase.cte`
        against a textual SELECT statement::

            stmt = stmt.columns(id=Integer, name=String).cte('st')

            stmt = select(sometable).where(sometable.c.id == stmt.c.id)

        :param \\*cols: A series of :class:`_expression.ColumnElement` objects,
         typically
         :class:`_schema.Column` objects from a :class:`_schema.Table`
         or ORM level
         column-mapped attributes, representing a set of columns that this
         textual string will SELECT from.

        :param \\**types: A mapping of string names to :class:`.TypeEngine`
         type objects indicating the datatypes to use for names that are
         SELECTed from the textual string.  Prefer to use the ``*cols``
         argument as it also indicates positional ordering.

        """
        selectable = util.preloaded.sql_selectable
        input_cols: List[NamedColumn[Any]] = [coercions.expect(roles.LabeledColumnExprRole, col) for col in cols]
        positional_input_cols = [ColumnClause(col.key, types.pop(col.key)) if col.key in types else col for col in input_cols]
        keyed_input_cols: List[NamedColumn[Any]] = [ColumnClause(key, type_) for key, type_ in types.items()]
        elem = selectable.TextualSelect.__new__(selectable.TextualSelect)
        elem._init(self, positional_input_cols + keyed_input_cols, positional=bool(positional_input_cols) and (not keyed_input_cols))
        return elem

    @property
    def type(self) -> TypeEngine[Any]:
        return type_api.NULLTYPE

    @property
    def comparator(self):
        return self.type.comparator_factory(self)

    def self_group(self, against=None):
        if against is operators.in_op:
            return Grouping(self)
        else:
            return self