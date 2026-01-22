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
class UpdateBase(roles.DMLRole, HasCTE, HasCompileState, DialectKWArgs, HasPrefixes, Generative, ExecutableReturnsRows, ClauseElement):
    """Form the base for ``INSERT``, ``UPDATE``, and ``DELETE`` statements."""
    __visit_name__ = 'update_base'
    _hints: util.immutabledict[Tuple[_DMLTableElement, str], str] = util.EMPTY_DICT
    named_with_column = False
    _label_style: SelectLabelStyle = SelectLabelStyle.LABEL_STYLE_DISAMBIGUATE_ONLY
    table: _DMLTableElement
    _return_defaults = False
    _return_defaults_columns: Optional[Tuple[_ColumnsClauseElement, ...]] = None
    _supplemental_returning: Optional[Tuple[_ColumnsClauseElement, ...]] = None
    _returning: Tuple[_ColumnsClauseElement, ...] = ()
    is_dml = True

    def _generate_fromclause_column_proxies(self, fromclause: FromClause) -> None:
        fromclause._columns._populate_separate_keys((col._make_proxy(fromclause) for col in self._all_selected_columns if is_column_element(col)))

    def params(self, *arg: Any, **kw: Any) -> NoReturn:
        """Set the parameters for the statement.

        This method raises ``NotImplementedError`` on the base class,
        and is overridden by :class:`.ValuesBase` to provide the
        SET/VALUES clause of UPDATE and INSERT.

        """
        raise NotImplementedError('params() is not supported for INSERT/UPDATE/DELETE statements. To set the values for an INSERT or UPDATE statement, use stmt.values(**parameters).')

    @_generative
    def with_dialect_options(self, **opt: Any) -> Self:
        """Add dialect options to this INSERT/UPDATE/DELETE object.

        e.g.::

            upd = table.update().dialect_options(mysql_limit=10)

        .. versionadded: 1.4 - this method supersedes the dialect options
           associated with the constructor.


        """
        self._validate_dialect_kwargs(opt)
        return self

    @_generative
    def return_defaults(self, *cols: _DMLColumnArgument, supplemental_cols: Optional[Iterable[_DMLColumnArgument]]=None, sort_by_parameter_order: bool=False) -> Self:
        """Make use of a :term:`RETURNING` clause for the purpose
        of fetching server-side expressions and defaults, for supporting
        backends only.

        .. deepalchemy::

            The :meth:`.UpdateBase.return_defaults` method is used by the ORM
            for its internal work in fetching newly generated primary key
            and server default values, in particular to provide the underyling
            implementation of the :paramref:`_orm.Mapper.eager_defaults`
            ORM feature as well as to allow RETURNING support with bulk
            ORM inserts.  Its behavior is fairly idiosyncratic
            and is not really intended for general use.  End users should
            stick with using :meth:`.UpdateBase.returning` in order to
            add RETURNING clauses to their INSERT, UPDATE and DELETE
            statements.

        Normally, a single row INSERT statement will automatically populate the
        :attr:`.CursorResult.inserted_primary_key` attribute when executed,
        which stores the primary key of the row that was just inserted in the
        form of a :class:`.Row` object with column names as named tuple keys
        (and the :attr:`.Row._mapping` view fully populated as well). The
        dialect in use chooses the strategy to use in order to populate this
        data; if it was generated using server-side defaults and / or SQL
        expressions, dialect-specific approaches such as ``cursor.lastrowid``
        or ``RETURNING`` are typically used to acquire the new primary key
        value.

        However, when the statement is modified by calling
        :meth:`.UpdateBase.return_defaults` before executing the statement,
        additional behaviors take place **only** for backends that support
        RETURNING and for :class:`.Table` objects that maintain the
        :paramref:`.Table.implicit_returning` parameter at its default value of
        ``True``. In these cases, when the :class:`.CursorResult` is returned
        from the statement's execution, not only will
        :attr:`.CursorResult.inserted_primary_key` be populated as always, the
        :attr:`.CursorResult.returned_defaults` attribute will also be
        populated with a :class:`.Row` named-tuple representing the full range
        of server generated
        values from that single row, including values for any columns that
        specify :paramref:`_schema.Column.server_default` or which make use of
        :paramref:`_schema.Column.default` using a SQL expression.

        When invoking INSERT statements with multiple rows using
        :ref:`insertmanyvalues <engine_insertmanyvalues>`, the
        :meth:`.UpdateBase.return_defaults` modifier will have the effect of
        the :attr:`_engine.CursorResult.inserted_primary_key_rows` and
        :attr:`_engine.CursorResult.returned_defaults_rows` attributes being
        fully populated with lists of :class:`.Row` objects representing newly
        inserted primary key values as well as newly inserted server generated
        values for each row inserted. The
        :attr:`.CursorResult.inserted_primary_key` and
        :attr:`.CursorResult.returned_defaults` attributes will also continue
        to be populated with the first row of these two collections.

        If the backend does not support RETURNING or the :class:`.Table` in use
        has disabled :paramref:`.Table.implicit_returning`, then no RETURNING
        clause is added and no additional data is fetched, however the
        INSERT, UPDATE or DELETE statement proceeds normally.

        E.g.::

            stmt = table.insert().values(data='newdata').return_defaults()

            result = connection.execute(stmt)

            server_created_at = result.returned_defaults['created_at']

        When used against an UPDATE statement
        :meth:`.UpdateBase.return_defaults` instead looks for columns that
        include :paramref:`_schema.Column.onupdate` or
        :paramref:`_schema.Column.server_onupdate` parameters assigned, when
        constructing the columns that will be included in the RETURNING clause
        by default if explicit columns were not specified. When used against a
        DELETE statement, no columns are included in RETURNING by default, they
        instead must be specified explicitly as there are no columns that
        normally change values when a DELETE statement proceeds.

        .. versionadded:: 2.0  :meth:`.UpdateBase.return_defaults` is supported
           for DELETE statements also and has been moved from
           :class:`.ValuesBase` to :class:`.UpdateBase`.

        The :meth:`.UpdateBase.return_defaults` method is mutually exclusive
        against the :meth:`.UpdateBase.returning` method and errors will be
        raised during the SQL compilation process if both are used at the same
        time on one statement. The RETURNING clause of the INSERT, UPDATE or
        DELETE statement is therefore controlled by only one of these methods
        at a time.

        The :meth:`.UpdateBase.return_defaults` method differs from
        :meth:`.UpdateBase.returning` in these ways:

        1. :meth:`.UpdateBase.return_defaults` method causes the
           :attr:`.CursorResult.returned_defaults` collection to be populated
           with the first row from the RETURNING result. This attribute is not
           populated when using :meth:`.UpdateBase.returning`.

        2. :meth:`.UpdateBase.return_defaults` is compatible with existing
           logic used to fetch auto-generated primary key values that are then
           populated into the :attr:`.CursorResult.inserted_primary_key`
           attribute. By contrast, using :meth:`.UpdateBase.returning` will
           have the effect of the :attr:`.CursorResult.inserted_primary_key`
           attribute being left unpopulated.

        3. :meth:`.UpdateBase.return_defaults` can be called against any
           backend. Backends that don't support RETURNING will skip the usage
           of the feature, rather than raising an exception, *unless*
           ``supplemental_cols`` is passed. The return value
           of :attr:`_engine.CursorResult.returned_defaults` will be ``None``
           for backends that don't support RETURNING or for which the target
           :class:`.Table` sets :paramref:`.Table.implicit_returning` to
           ``False``.

        4. An INSERT statement invoked with executemany() is supported if the
           backend database driver supports the
           :ref:`insertmanyvalues <engine_insertmanyvalues>`
           feature which is now supported by most SQLAlchemy-included backends.
           When executemany is used, the
           :attr:`_engine.CursorResult.returned_defaults_rows` and
           :attr:`_engine.CursorResult.inserted_primary_key_rows` accessors
           will return the inserted defaults and primary keys.

           .. versionadded:: 1.4 Added
              :attr:`_engine.CursorResult.returned_defaults_rows` and
              :attr:`_engine.CursorResult.inserted_primary_key_rows` accessors.
              In version 2.0, the underlying implementation which fetches and
              populates the data for these attributes was generalized to be
              supported by most backends, whereas in 1.4 they were only
              supported by the ``psycopg2`` driver.


        :param cols: optional list of column key names or
         :class:`_schema.Column` that acts as a filter for those columns that
         will be fetched.
        :param supplemental_cols: optional list of RETURNING expressions,
          in the same form as one would pass to the
          :meth:`.UpdateBase.returning` method. When present, the additional
          columns will be included in the RETURNING clause, and the
          :class:`.CursorResult` object will be "rewound" when returned, so
          that methods like :meth:`.CursorResult.all` will return new rows
          mostly as though the statement used :meth:`.UpdateBase.returning`
          directly. However, unlike when using :meth:`.UpdateBase.returning`
          directly, the **order of the columns is undefined**, so can only be
          targeted using names or :attr:`.Row._mapping` keys; they cannot
          reliably be targeted positionally.

          .. versionadded:: 2.0

        :param sort_by_parameter_order: for a batch INSERT that is being
         executed against multiple parameter sets, organize the results of
         RETURNING so that the returned rows correspond to the order of
         parameter sets passed in.  This applies only to an :term:`executemany`
         execution for supporting dialects and typically makes use of the
         :term:`insertmanyvalues` feature.

         .. versionadded:: 2.0.10

         .. seealso::

            :ref:`engine_insertmanyvalues_returning_order` - background on
            sorting of RETURNING rows for bulk INSERT

        .. seealso::

            :meth:`.UpdateBase.returning`

            :attr:`_engine.CursorResult.returned_defaults`

            :attr:`_engine.CursorResult.returned_defaults_rows`

            :attr:`_engine.CursorResult.inserted_primary_key`

            :attr:`_engine.CursorResult.inserted_primary_key_rows`

        """
        if self._return_defaults:
            if self._return_defaults_columns and cols:
                self._return_defaults_columns = tuple(util.OrderedSet(self._return_defaults_columns).union((coercions.expect(roles.ColumnsClauseRole, c) for c in cols)))
            else:
                self._return_defaults_columns = ()
        else:
            self._return_defaults_columns = tuple((coercions.expect(roles.ColumnsClauseRole, c) for c in cols))
        self._return_defaults = True
        if sort_by_parameter_order:
            if not self.is_insert:
                raise exc.ArgumentError("The 'sort_by_parameter_order' argument to return_defaults() only applies to INSERT statements")
            self._sort_by_parameter_order = True
        if supplemental_cols:
            supplemental_col_tup = (coercions.expect(roles.ColumnsClauseRole, c) for c in supplemental_cols)
            if self._supplemental_returning is None:
                self._supplemental_returning = tuple(util.unique_list(supplemental_col_tup))
            else:
                self._supplemental_returning = tuple(util.unique_list(self._supplemental_returning + tuple(supplemental_col_tup)))
        return self

    @_generative
    def returning(self, *cols: _ColumnsClauseArgument[Any], sort_by_parameter_order: bool=False, **__kw: Any) -> UpdateBase:
        """Add a :term:`RETURNING` or equivalent clause to this statement.

        e.g.:

        .. sourcecode:: pycon+sql

            >>> stmt = (
            ...     table.update()
            ...     .where(table.c.data == "value")
            ...     .values(status="X")
            ...     .returning(table.c.server_flag, table.c.updated_timestamp)
            ... )
            >>> print(stmt)
            {printsql}UPDATE some_table SET status=:status
            WHERE some_table.data = :data_1
            RETURNING some_table.server_flag, some_table.updated_timestamp

        The method may be invoked multiple times to add new entries to the
        list of expressions to be returned.

        .. versionadded:: 1.4.0b2 The method may be invoked multiple times to
         add new entries to the list of expressions to be returned.

        The given collection of column expressions should be derived from the
        table that is the target of the INSERT, UPDATE, or DELETE.  While
        :class:`_schema.Column` objects are typical, the elements can also be
        expressions:

        .. sourcecode:: pycon+sql

            >>> stmt = table.insert().returning(
            ...     (table.c.first_name + " " + table.c.last_name).label("fullname")
            ... )
            >>> print(stmt)
            {printsql}INSERT INTO some_table (first_name, last_name)
            VALUES (:first_name, :last_name)
            RETURNING some_table.first_name || :first_name_1 || some_table.last_name AS fullname

        Upon compilation, a RETURNING clause, or database equivalent,
        will be rendered within the statement.   For INSERT and UPDATE,
        the values are the newly inserted/updated values.  For DELETE,
        the values are those of the rows which were deleted.

        Upon execution, the values of the columns to be returned are made
        available via the result set and can be iterated using
        :meth:`_engine.CursorResult.fetchone` and similar.
        For DBAPIs which do not
        natively support returning values (i.e. cx_oracle), SQLAlchemy will
        approximate this behavior at the result level so that a reasonable
        amount of behavioral neutrality is provided.

        Note that not all databases/DBAPIs
        support RETURNING.   For those backends with no support,
        an exception is raised upon compilation and/or execution.
        For those who do support it, the functionality across backends
        varies greatly, including restrictions on executemany()
        and other statements which return multiple rows. Please
        read the documentation notes for the database in use in
        order to determine the availability of RETURNING.

        :param \\*cols: series of columns, SQL expressions, or whole tables
         entities to be returned.
        :param sort_by_parameter_order: for a batch INSERT that is being
         executed against multiple parameter sets, organize the results of
         RETURNING so that the returned rows correspond to the order of
         parameter sets passed in.  This applies only to an :term:`executemany`
         execution for supporting dialects and typically makes use of the
         :term:`insertmanyvalues` feature.

         .. versionadded:: 2.0.10

         .. seealso::

            :ref:`engine_insertmanyvalues_returning_order` - background on
            sorting of RETURNING rows for bulk INSERT (Core level discussion)

            :ref:`orm_queryguide_bulk_insert_returning_ordered` - example of
            use with :ref:`orm_queryguide_bulk_insert` (ORM level discussion)

        .. seealso::

          :meth:`.UpdateBase.return_defaults` - an alternative method tailored
          towards efficient fetching of server-side defaults and triggers
          for single-row INSERTs or UPDATEs.

          :ref:`tutorial_insert_returning` - in the :ref:`unified_tutorial`

        """
        if __kw:
            raise _unexpected_kw('UpdateBase.returning()', __kw)
        if self._return_defaults:
            raise exc.InvalidRequestError('return_defaults() is already configured on this statement')
        self._returning += tuple((coercions.expect(roles.ColumnsClauseRole, c) for c in cols))
        if sort_by_parameter_order:
            if not self.is_insert:
                raise exc.ArgumentError("The 'sort_by_parameter_order' argument to returning() only applies to INSERT statements")
            self._sort_by_parameter_order = True
        return self

    def corresponding_column(self, column: KeyedColumnElement[Any], require_embedded: bool=False) -> Optional[ColumnElement[Any]]:
        return self.exported_columns.corresponding_column(column, require_embedded=require_embedded)

    @util.ro_memoized_property
    def _all_selected_columns(self) -> _SelectIterable:
        return [c for c in _select_iterables(self._returning)]

    @util.ro_memoized_property
    def exported_columns(self) -> ReadOnlyColumnCollection[Optional[str], ColumnElement[Any]]:
        """Return the RETURNING columns as a column collection for this
        statement.

        .. versionadded:: 1.4

        """
        return ColumnCollection(((c.key, c) for c in self._all_selected_columns if is_column_element(c))).as_readonly()

    @_generative
    def with_hint(self, text: str, selectable: Optional[_DMLTableArgument]=None, dialect_name: str='*') -> Self:
        """Add a table hint for a single table to this
        INSERT/UPDATE/DELETE statement.

        .. note::

         :meth:`.UpdateBase.with_hint` currently applies only to
         Microsoft SQL Server.  For MySQL INSERT/UPDATE/DELETE hints, use
         :meth:`.UpdateBase.prefix_with`.

        The text of the hint is rendered in the appropriate
        location for the database backend in use, relative
        to the :class:`_schema.Table` that is the subject of this
        statement, or optionally to that of the given
        :class:`_schema.Table` passed as the ``selectable`` argument.

        The ``dialect_name`` option will limit the rendering of a particular
        hint to a particular backend. Such as, to add a hint
        that only takes effect for SQL Server::

            mytable.insert().with_hint("WITH (PAGLOCK)", dialect_name="mssql")

        :param text: Text of the hint.
        :param selectable: optional :class:`_schema.Table` that specifies
         an element of the FROM clause within an UPDATE or DELETE
         to be the subject of the hint - applies only to certain backends.
        :param dialect_name: defaults to ``*``, if specified as the name
         of a particular dialect, will apply these hints only when
         that dialect is in use.
        """
        if selectable is None:
            selectable = self.table
        else:
            selectable = coercions.expect(roles.DMLTableRole, selectable)
        self._hints = self._hints.union({(selectable, dialect_name): text})
        return self

    @property
    def entity_description(self) -> Dict[str, Any]:
        """Return a :term:`plugin-enabled` description of the table and/or
        entity which this DML construct is operating against.

        This attribute is generally useful when using the ORM, as an
        extended structure which includes information about mapped
        entities is returned.  The section :ref:`queryguide_inspection`
        contains more background.

        For a Core statement, the structure returned by this accessor
        is derived from the :attr:`.UpdateBase.table` attribute, and
        refers to the :class:`.Table` being inserted, updated, or deleted::

            >>> stmt = insert(user_table)
            >>> stmt.entity_description
            {
                "name": "user_table",
                "table": Table("user_table", ...)
            }

        .. versionadded:: 1.4.33

        .. seealso::

            :attr:`.UpdateBase.returning_column_descriptions`

            :attr:`.Select.column_descriptions` - entity information for
            a :func:`.select` construct

            :ref:`queryguide_inspection` - ORM background

        """
        meth = DMLState.get_plugin_class(self).get_entity_description
        return meth(self)

    @property
    def returning_column_descriptions(self) -> List[Dict[str, Any]]:
        """Return a :term:`plugin-enabled` description of the columns
        which this DML construct is RETURNING against, in other words
        the expressions established as part of :meth:`.UpdateBase.returning`.

        This attribute is generally useful when using the ORM, as an
        extended structure which includes information about mapped
        entities is returned.  The section :ref:`queryguide_inspection`
        contains more background.

        For a Core statement, the structure returned by this accessor is
        derived from the same objects that are returned by the
        :attr:`.UpdateBase.exported_columns` accessor::

            >>> stmt = insert(user_table).returning(user_table.c.id, user_table.c.name)
            >>> stmt.entity_description
            [
                {
                    "name": "id",
                    "type": Integer,
                    "expr": Column("id", Integer(), table=<user>, ...)
                },
                {
                    "name": "name",
                    "type": String(),
                    "expr": Column("name", String(), table=<user>, ...)
                },
            ]

        .. versionadded:: 1.4.33

        .. seealso::

            :attr:`.UpdateBase.entity_description`

            :attr:`.Select.column_descriptions` - entity information for
            a :func:`.select` construct

            :ref:`queryguide_inspection` - ORM background

        """
        meth = DMLState.get_plugin_class(self).get_returning_column_descriptions
        return meth(self)