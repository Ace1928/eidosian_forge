from __future__ import annotations
import datetime
import decimal
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import annotation
from . import coercions
from . import operators
from . import roles
from . import schema
from . import sqltypes
from . import type_api
from . import util as sqlutil
from ._typing import is_table_value_type
from .base import _entity_namespace
from .base import ColumnCollection
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .elements import _type_from_args
from .elements import BinaryExpression
from .elements import BindParameter
from .elements import Cast
from .elements import ClauseList
from .elements import ColumnElement
from .elements import Extract
from .elements import FunctionFilter
from .elements import Grouping
from .elements import literal_column
from .elements import NamedColumn
from .elements import Over
from .elements import WithinGroup
from .selectable import FromClause
from .selectable import Select
from .selectable import TableValuedAlias
from .sqltypes import TableValueType
from .type_api import TypeEngine
from .visitors import InternalTraversal
from .. import util
class FunctionElement(Executable, ColumnElement[_T], FromClause, Generative):
    """Base for SQL function-oriented constructs.

    This is a `generic type <https://peps.python.org/pep-0484/#generics>`_,
    meaning that type checkers and IDEs can be instructed on the types to
    expect in a :class:`_engine.Result` for this function. See
    :class:`.GenericFunction` for an example of how this is done.

    .. seealso::

        :ref:`tutorial_functions` - in the :ref:`unified_tutorial`

        :class:`.Function` - named SQL function.

        :data:`.func` - namespace which produces registered or ad-hoc
        :class:`.Function` instances.

        :class:`.GenericFunction` - allows creation of registered function
        types.

    """
    _traverse_internals = [('clause_expr', InternalTraversal.dp_clauseelement), ('_with_ordinality', InternalTraversal.dp_boolean), ('_table_value_type', InternalTraversal.dp_has_cache_key)]
    packagenames: Tuple[str, ...] = ()
    _has_args = False
    _with_ordinality = False
    _table_value_type: Optional[TableValueType] = None
    primary_key: Any
    _is_clone_of: Any
    clause_expr: Grouping[Any]

    def __init__(self, *clauses: _ColumnExpressionOrLiteralArgument[Any]):
        """Construct a :class:`.FunctionElement`.

        :param \\*clauses: list of column expressions that form the arguments
         of the SQL function call.

        :param \\**kwargs:  additional kwargs are typically consumed by
         subclasses.

        .. seealso::

            :data:`.func`

            :class:`.Function`

        """
        args: Sequence[_ColumnExpressionArgument[Any]] = [coercions.expect(roles.ExpressionElementRole, c, name=getattr(self, 'name', None), apply_propagate_attrs=self) for c in clauses]
        self._has_args = self._has_args or bool(args)
        self.clause_expr = Grouping(ClauseList(*args, operator=operators.comma_op, group_contents=True))
    _non_anon_label = None

    @property
    def _proxy_key(self) -> Any:
        return super()._proxy_key or getattr(self, 'name', None)

    def _execute_on_connection(self, connection: Connection, distilled_params: _CoreMultiExecuteParams, execution_options: CoreExecuteOptionsParameter) -> CursorResult[Any]:
        return connection._execute_function(self, distilled_params, execution_options)

    def scalar_table_valued(self, name: str, type_: Optional[_TypeEngineArgument[_T]]=None) -> ScalarFunctionColumn[_T]:
        """Return a column expression that's against this
        :class:`_functions.FunctionElement` as a scalar
        table-valued expression.

        The returned expression is similar to that returned by a single column
        accessed off of a :meth:`_functions.FunctionElement.table_valued`
        construct, except no FROM clause is generated; the function is rendered
        in the similar way as a scalar subquery.

        E.g.:

        .. sourcecode:: pycon+sql

            >>> from sqlalchemy import func, select
            >>> fn = func.jsonb_each("{'k', 'v'}").scalar_table_valued("key")
            >>> print(select(fn))
            {printsql}SELECT (jsonb_each(:jsonb_each_1)).key

        .. versionadded:: 1.4.0b2

        .. seealso::

            :meth:`_functions.FunctionElement.table_valued`

            :meth:`_functions.FunctionElement.alias`

            :meth:`_functions.FunctionElement.column_valued`

        """
        return ScalarFunctionColumn(self, name, type_)

    def table_valued(self, *expr: _ColumnExpressionOrStrLabelArgument[Any], **kw: Any) -> TableValuedAlias:
        """Return a :class:`_sql.TableValuedAlias` representation of this
        :class:`_functions.FunctionElement` with table-valued expressions added.

        e.g.:

        .. sourcecode:: pycon+sql

            >>> fn = (
            ...     func.generate_series(1, 5).
            ...     table_valued("value", "start", "stop", "step")
            ... )

            >>> print(select(fn))
            {printsql}SELECT anon_1.value, anon_1.start, anon_1.stop, anon_1.step
            FROM generate_series(:generate_series_1, :generate_series_2) AS anon_1{stop}

            >>> print(select(fn.c.value, fn.c.stop).where(fn.c.value > 2))
            {printsql}SELECT anon_1.value, anon_1.stop
            FROM generate_series(:generate_series_1, :generate_series_2) AS anon_1
            WHERE anon_1.value > :value_1{stop}

        A WITH ORDINALITY expression may be generated by passing the keyword
        argument "with_ordinality":

        .. sourcecode:: pycon+sql

            >>> fn = func.generate_series(4, 1, -1).table_valued("gen", with_ordinality="ordinality")
            >>> print(select(fn))
            {printsql}SELECT anon_1.gen, anon_1.ordinality
            FROM generate_series(:generate_series_1, :generate_series_2, :generate_series_3) WITH ORDINALITY AS anon_1

        :param \\*expr: A series of string column names that will be added to the
         ``.c`` collection of the resulting :class:`_sql.TableValuedAlias`
         construct as columns.  :func:`_sql.column` objects with or without
         datatypes may also be used.

        :param name: optional name to assign to the alias name that's generated.
         If omitted, a unique anonymizing name is used.

        :param with_ordinality: string name that when present results in the
         ``WITH ORDINALITY`` clause being added to the alias, and the given
         string name will be added as a column to the .c collection
         of the resulting :class:`_sql.TableValuedAlias`.

        :param joins_implicitly: when True, the table valued function may be
         used in the FROM clause without any explicit JOIN to other tables
         in the SQL query, and no "cartesian product" warning will be generated.
         May be useful for SQL functions such as ``func.json_each()``.

         .. versionadded:: 1.4.33

        .. versionadded:: 1.4.0b2


        .. seealso::

            :ref:`tutorial_functions_table_valued` - in the :ref:`unified_tutorial`

            :ref:`postgresql_table_valued` - in the :ref:`postgresql_toplevel` documentation

            :meth:`_functions.FunctionElement.scalar_table_valued` - variant of
            :meth:`_functions.FunctionElement.table_valued` which delivers the
            complete table valued expression as a scalar column expression

            :meth:`_functions.FunctionElement.column_valued`

            :meth:`_sql.TableValuedAlias.render_derived` - renders the alias
            using a derived column clause, e.g. ``AS name(col1, col2, ...)``

        """
        new_func = self._generate()
        with_ordinality = kw.pop('with_ordinality', None)
        joins_implicitly = kw.pop('joins_implicitly', None)
        name = kw.pop('name', None)
        if with_ordinality:
            expr += (with_ordinality,)
            new_func._with_ordinality = True
        new_func.type = new_func._table_value_type = TableValueType(*expr)
        return new_func.alias(name=name, joins_implicitly=joins_implicitly)

    def column_valued(self, name: Optional[str]=None, joins_implicitly: bool=False) -> TableValuedColumn[_T]:
        """Return this :class:`_functions.FunctionElement` as a column expression that
        selects from itself as a FROM clause.

        E.g.:

        .. sourcecode:: pycon+sql

            >>> from sqlalchemy import select, func
            >>> gs = func.generate_series(1, 5, -1).column_valued()
            >>> print(select(gs))
            {printsql}SELECT anon_1
            FROM generate_series(:generate_series_1, :generate_series_2, :generate_series_3) AS anon_1

        This is shorthand for::

            gs = func.generate_series(1, 5, -1).alias().column

        :param name: optional name to assign to the alias name that's generated.
         If omitted, a unique anonymizing name is used.

        :param joins_implicitly: when True, the "table" portion of the column
         valued function may be a member of the FROM clause without any
         explicit JOIN to other tables in the SQL query, and no "cartesian
         product" warning will be generated. May be useful for SQL functions
         such as ``func.json_array_elements()``.

         .. versionadded:: 1.4.46

        .. seealso::

            :ref:`tutorial_functions_column_valued` - in the :ref:`unified_tutorial`

            :ref:`postgresql_column_valued` - in the :ref:`postgresql_toplevel` documentation

            :meth:`_functions.FunctionElement.table_valued`

        """
        return self.alias(name=name, joins_implicitly=joins_implicitly).column

    @util.ro_non_memoized_property
    def columns(self) -> ColumnCollection[str, KeyedColumnElement[Any]]:
        """The set of columns exported by this :class:`.FunctionElement`.

        This is a placeholder collection that allows the function to be
        placed in the FROM clause of a statement:

        .. sourcecode:: pycon+sql

            >>> from sqlalchemy import column, select, func
            >>> stmt = select(column('x'), column('y')).select_from(func.myfunction())
            >>> print(stmt)
            {printsql}SELECT x, y FROM myfunction()

        The above form is a legacy feature that is now superseded by the
        fully capable :meth:`_functions.FunctionElement.table_valued`
        method; see that method for details.

        .. seealso::

            :meth:`_functions.FunctionElement.table_valued` - generates table-valued
            SQL function expressions.

        """
        return self.c

    @util.ro_memoized_property
    def c(self) -> ColumnCollection[str, KeyedColumnElement[Any]]:
        """synonym for :attr:`.FunctionElement.columns`."""
        return ColumnCollection(columns=[(col.key, col) for col in self._all_selected_columns])

    @property
    def _all_selected_columns(self) -> Sequence[KeyedColumnElement[Any]]:
        if is_table_value_type(self.type):
            cols = cast('Sequence[KeyedColumnElement[Any]]', self.type._elements)
        else:
            cols = [self.label(None)]
        return cols

    @property
    def exported_columns(self) -> ColumnCollection[str, KeyedColumnElement[Any]]:
        return self.columns

    @HasMemoized.memoized_attribute
    def clauses(self) -> ClauseList:
        """Return the underlying :class:`.ClauseList` which contains
        the arguments for this :class:`.FunctionElement`.

        """
        return cast(ClauseList, self.clause_expr.element)

    def over(self, *, partition_by: Optional[_ByArgument]=None, order_by: Optional[_ByArgument]=None, rows: Optional[Tuple[Optional[int], Optional[int]]]=None, range_: Optional[Tuple[Optional[int], Optional[int]]]=None) -> Over[_T]:
        """Produce an OVER clause against this function.

        Used against aggregate or so-called "window" functions,
        for database backends that support window functions.

        The expression::

            func.row_number().over(order_by='x')

        is shorthand for::

            from sqlalchemy import over
            over(func.row_number(), order_by='x')

        See :func:`_expression.over` for a full description.

        .. seealso::

            :func:`_expression.over`

            :ref:`tutorial_window_functions` - in the :ref:`unified_tutorial`

        """
        return Over(self, partition_by=partition_by, order_by=order_by, rows=rows, range_=range_)

    def within_group(self, *order_by: _ColumnExpressionArgument[Any]) -> WithinGroup[_T]:
        """Produce a WITHIN GROUP (ORDER BY expr) clause against this function.

        Used against so-called "ordered set aggregate" and "hypothetical
        set aggregate" functions, including :class:`.percentile_cont`,
        :class:`.rank`, :class:`.dense_rank`, etc.

        See :func:`_expression.within_group` for a full description.

        .. seealso::

            :ref:`tutorial_functions_within_group` -
            in the :ref:`unified_tutorial`


        """
        return WithinGroup(self, *order_by)

    @overload
    def filter(self) -> Self:
        ...

    @overload
    def filter(self, __criterion0: _ColumnExpressionArgument[bool], *criterion: _ColumnExpressionArgument[bool]) -> FunctionFilter[_T]:
        ...

    def filter(self, *criterion: _ColumnExpressionArgument[bool]) -> Union[Self, FunctionFilter[_T]]:
        """Produce a FILTER clause against this function.

        Used against aggregate and window functions,
        for database backends that support the "FILTER" clause.

        The expression::

            func.count(1).filter(True)

        is shorthand for::

            from sqlalchemy import funcfilter
            funcfilter(func.count(1), True)

        .. seealso::

            :ref:`tutorial_functions_within_group` -
            in the :ref:`unified_tutorial`

            :class:`.FunctionFilter`

            :func:`.funcfilter`


        """
        if not criterion:
            return self
        return FunctionFilter(self, *criterion)

    def as_comparison(self, left_index: int, right_index: int) -> FunctionAsBinary:
        """Interpret this expression as a boolean comparison between two
        values.

        This method is used for an ORM use case described at
        :ref:`relationship_custom_operator_sql_function`.

        A hypothetical SQL function "is_equal()" which compares to values
        for equality would be written in the Core expression language as::

            expr = func.is_equal("a", "b")

        If "is_equal()" above is comparing "a" and "b" for equality, the
        :meth:`.FunctionElement.as_comparison` method would be invoked as::

            expr = func.is_equal("a", "b").as_comparison(1, 2)

        Where above, the integer value "1" refers to the first argument of the
        "is_equal()" function and the integer value "2" refers to the second.

        This would create a :class:`.BinaryExpression` that is equivalent to::

            BinaryExpression("a", "b", operator=op.eq)

        However, at the SQL level it would still render as
        "is_equal('a', 'b')".

        The ORM, when it loads a related object or collection, needs to be able
        to manipulate the "left" and "right" sides of the ON clause of a JOIN
        expression. The purpose of this method is to provide a SQL function
        construct that can also supply this information to the ORM, when used
        with the :paramref:`_orm.relationship.primaryjoin` parameter. The
        return value is a containment object called :class:`.FunctionAsBinary`.

        An ORM example is as follows::

            class Venue(Base):
                __tablename__ = 'venue'
                id = Column(Integer, primary_key=True)
                name = Column(String)

                descendants = relationship(
                    "Venue",
                    primaryjoin=func.instr(
                        remote(foreign(name)), name + "/"
                    ).as_comparison(1, 2) == 1,
                    viewonly=True,
                    order_by=name
                )

        Above, the "Venue" class can load descendant "Venue" objects by
        determining if the name of the parent Venue is contained within the
        start of the hypothetical descendant value's name, e.g. "parent1" would
        match up to "parent1/child1", but not to "parent2/child1".

        Possible use cases include the "materialized path" example given above,
        as well as making use of special SQL functions such as geometric
        functions to create join conditions.

        :param left_index: the integer 1-based index of the function argument
         that serves as the "left" side of the expression.
        :param right_index: the integer 1-based index of the function argument
         that serves as the "right" side of the expression.

        .. versionadded:: 1.3

        .. seealso::

            :ref:`relationship_custom_operator_sql_function` -
            example use within the ORM

        """
        return FunctionAsBinary(self, left_index, right_index)

    @property
    def _from_objects(self) -> Any:
        return self.clauses._from_objects

    def within_group_type(self, within_group: WithinGroup[_S]) -> Optional[TypeEngine[_S]]:
        """For types that define their return type as based on the criteria
        within a WITHIN GROUP (ORDER BY) expression, called by the
        :class:`.WithinGroup` construct.

        Returns None by default, in which case the function's normal ``.type``
        is used.

        """
        return None

    def alias(self, name: Optional[str]=None, joins_implicitly: bool=False) -> TableValuedAlias:
        """Produce a :class:`_expression.Alias` construct against this
        :class:`.FunctionElement`.

        .. tip::

            The :meth:`_functions.FunctionElement.alias` method is part of the
            mechanism by which "table valued" SQL functions are created.
            However, most use cases are covered by higher level methods on
            :class:`_functions.FunctionElement` including
            :meth:`_functions.FunctionElement.table_valued`, and
            :meth:`_functions.FunctionElement.column_valued`.

        This construct wraps the function in a named alias which
        is suitable for the FROM clause, in the style accepted for example
        by PostgreSQL.  A column expression is also provided using the
        special ``.column`` attribute, which may
        be used to refer to the output of the function as a scalar value
        in the columns or where clause, for a backend such as PostgreSQL.

        For a full table-valued expression, use the
        :meth:`_functions.FunctionElement.table_valued` method first to
        establish named columns.

        e.g.:

        .. sourcecode:: pycon+sql

            >>> from sqlalchemy import func, select, column
            >>> data_view = func.unnest([1, 2, 3]).alias("data_view")
            >>> print(select(data_view.column))
            {printsql}SELECT data_view
            FROM unnest(:unnest_1) AS data_view

        The :meth:`_functions.FunctionElement.column_valued` method provides
        a shortcut for the above pattern:

        .. sourcecode:: pycon+sql

            >>> data_view = func.unnest([1, 2, 3]).column_valued("data_view")
            >>> print(select(data_view))
            {printsql}SELECT data_view
            FROM unnest(:unnest_1) AS data_view

        .. versionadded:: 1.4.0b2  Added the ``.column`` accessor

        :param name: alias name, will be rendered as ``AS <name>`` in the
         FROM clause

        :param joins_implicitly: when True, the table valued function may be
         used in the FROM clause without any explicit JOIN to other tables
         in the SQL query, and no "cartesian product" warning will be
         generated.  May be useful for SQL functions such as
         ``func.json_each()``.

         .. versionadded:: 1.4.33

        .. seealso::

            :ref:`tutorial_functions_table_valued` -
            in the :ref:`unified_tutorial`

            :meth:`_functions.FunctionElement.table_valued`

            :meth:`_functions.FunctionElement.scalar_table_valued`

            :meth:`_functions.FunctionElement.column_valued`


        """
        return TableValuedAlias._construct(self, name=name, table_value_type=self.type, joins_implicitly=joins_implicitly)

    def select(self) -> Select[Tuple[_T]]:
        """Produce a :func:`_expression.select` construct
        against this :class:`.FunctionElement`.

        This is shorthand for::

            s = select(function_element)

        """
        s: Select[Any] = Select(self)
        if self._execution_options:
            s = s.execution_options(**self._execution_options)
        return s

    def _bind_param(self, operator: OperatorType, obj: Any, type_: Optional[TypeEngine[_T]]=None, expanding: bool=False, **kw: Any) -> BindParameter[_T]:
        return BindParameter(None, obj, _compared_to_operator=operator, _compared_to_type=self.type, unique=True, type_=type_, expanding=expanding, **kw)

    def self_group(self, against: Optional[OperatorType]=None) -> ClauseElement:
        if against is operators.getitem and isinstance(self.type, sqltypes.ARRAY):
            return Grouping(self)
        else:
            return super().self_group(against=against)

    @property
    def entity_namespace(self) -> _EntityNamespace:
        """overrides FromClause.entity_namespace as functions are generally
        column expressions and not FromClauses.

        """
        return _entity_namespace(self.clause_expr)