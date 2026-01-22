from __future__ import annotations
import collections
import functools
import operator
import typing
from typing import Any
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .result import IteratorResult
from .result import MergedResult
from .result import Result
from .result import ResultMetaData
from .result import SimpleResultMetaData
from .result import tuplegetter
from .row import Row
from .. import exc
from .. import util
from ..sql import elements
from ..sql import sqltypes
from ..sql import util as sql_util
from ..sql.base import _generative
from ..sql.compiler import ResultColumnsEntry
from ..sql.compiler import RM_NAME
from ..sql.compiler import RM_OBJECTS
from ..sql.compiler import RM_RENDERED_NAME
from ..sql.compiler import RM_TYPE
from ..sql.type_api import TypeEngine
from ..util import compat
from ..util.typing import Literal
from ..util.typing import Self
class CursorResult(Result[_T]):
    """A Result that is representing state from a DBAPI cursor.

    .. versionchanged:: 1.4  The :class:`.CursorResult``
       class replaces the previous :class:`.ResultProxy` interface.
       This classes are based on the :class:`.Result` calling API
       which provides an updated usage model and calling facade for
       SQLAlchemy Core and SQLAlchemy ORM.

    Returns database rows via the :class:`.Row` class, which provides
    additional API features and behaviors on top of the raw data returned by
    the DBAPI.   Through the use of filters such as the :meth:`.Result.scalars`
    method, other kinds of objects may also be returned.

    .. seealso::

        :ref:`tutorial_selecting_data` - introductory material for accessing
        :class:`_engine.CursorResult` and :class:`.Row` objects.

    """
    __slots__ = ('context', 'dialect', 'cursor', 'cursor_strategy', '_echo', 'connection')
    _metadata: Union[CursorResultMetaData, _NoResultMetaData]
    _no_result_metadata = _NO_RESULT_METADATA
    _soft_closed: bool = False
    closed: bool = False
    _is_cursor = True
    context: DefaultExecutionContext
    dialect: Dialect
    cursor_strategy: ResultFetchStrategy
    connection: Connection

    def __init__(self, context: DefaultExecutionContext, cursor_strategy: ResultFetchStrategy, cursor_description: Optional[_DBAPICursorDescription]):
        self.context = context
        self.dialect = context.dialect
        self.cursor = context.cursor
        self.cursor_strategy = cursor_strategy
        self.connection = context.root_connection
        self._echo = echo = self.connection._echo and context.engine._should_log_debug()
        if cursor_description is not None:
            metadata = self._init_metadata(context, cursor_description)
            _make_row: Any
            _make_row = functools.partial(Row, metadata, metadata._effective_processors, metadata._key_to_index)
            if context._num_sentinel_cols:
                sentinel_filter = operator.itemgetter(slice(-context._num_sentinel_cols))

                def _sliced_row(raw_data):
                    return _make_row(sentinel_filter(raw_data))
                sliced_row = _sliced_row
            else:
                sliced_row = _make_row
            if echo:
                log = self.context.connection._log_debug

                def _log_row(row):
                    log('Row %r', sql_util._repr_row(row))
                    return row
                self._row_logging_fn = _log_row

                def _make_row_2(row):
                    return _log_row(sliced_row(row))
                make_row = _make_row_2
            else:
                make_row = sliced_row
            self._set_memoized_attribute('_row_getter', make_row)
        else:
            assert context._num_sentinel_cols == 0
            self._metadata = self._no_result_metadata

    def _init_metadata(self, context, cursor_description):
        if context.compiled:
            compiled = context.compiled
            if compiled._cached_metadata:
                metadata = compiled._cached_metadata
            else:
                metadata = CursorResultMetaData(self, cursor_description)
                if metadata._safe_for_cache:
                    compiled._cached_metadata = metadata
            if not context.execution_options.get('_result_disable_adapt_to_context', False) and compiled._result_columns and (context.cache_hit is context.dialect.CACHE_HIT) and (compiled.statement is not context.invoked_statement):
                metadata = metadata._adapt_to_context(context)
            self._metadata = metadata
        else:
            self._metadata = metadata = CursorResultMetaData(self, cursor_description)
        if self._echo:
            context.connection._log_debug('Col %r', tuple((x[0] for x in cursor_description)))
        return metadata

    def _soft_close(self, hard=False):
        """Soft close this :class:`_engine.CursorResult`.

        This releases all DBAPI cursor resources, but leaves the
        CursorResult "open" from a semantic perspective, meaning the
        fetchXXX() methods will continue to return empty results.

        This method is called automatically when:

        * all result rows are exhausted using the fetchXXX() methods.
        * cursor.description is None.

        This method is **not public**, but is documented in order to clarify
        the "autoclose" process used.

        .. seealso::

            :meth:`_engine.CursorResult.close`


        """
        if not hard and self._soft_closed or (hard and self.closed):
            return
        if hard:
            self.closed = True
            self.cursor_strategy.hard_close(self, self.cursor)
        else:
            self.cursor_strategy.soft_close(self, self.cursor)
        if not self._soft_closed:
            cursor = self.cursor
            self.cursor = None
            self.connection._safe_close_cursor(cursor)
            self._soft_closed = True

    @property
    def inserted_primary_key_rows(self):
        """Return the value of
        :attr:`_engine.CursorResult.inserted_primary_key`
        as a row contained within a list; some dialects may support a
        multiple row form as well.

        .. note:: As indicated below, in current SQLAlchemy versions this
           accessor is only useful beyond what's already supplied by
           :attr:`_engine.CursorResult.inserted_primary_key` when using the
           :ref:`postgresql_psycopg2` dialect.   Future versions hope to
           generalize this feature to more dialects.

        This accessor is added to support dialects that offer the feature
        that is currently implemented by the :ref:`psycopg2_executemany_mode`
        feature, currently **only the psycopg2 dialect**, which provides
        for many rows to be INSERTed at once while still retaining the
        behavior of being able to return server-generated primary key values.

        * **When using the psycopg2 dialect, or other dialects that may support
          "fast executemany" style inserts in upcoming releases** : When
          invoking an INSERT statement while passing a list of rows as the
          second argument to :meth:`_engine.Connection.execute`, this accessor
          will then provide a list of rows, where each row contains the primary
          key value for each row that was INSERTed.

        * **When using all other dialects / backends that don't yet support
          this feature**: This accessor is only useful for **single row INSERT
          statements**, and returns the same information as that of the
          :attr:`_engine.CursorResult.inserted_primary_key` within a
          single-element list. When an INSERT statement is executed in
          conjunction with a list of rows to be INSERTed, the list will contain
          one row per row inserted in the statement, however it will contain
          ``None`` for any server-generated values.

        Future releases of SQLAlchemy will further generalize the
        "fast execution helper" feature of psycopg2 to suit other dialects,
        thus allowing this accessor to be of more general use.

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_engine.CursorResult.inserted_primary_key`

        """
        if not self.context.compiled:
            raise exc.InvalidRequestError('Statement is not a compiled expression construct.')
        elif not self.context.isinsert:
            raise exc.InvalidRequestError('Statement is not an insert() expression construct.')
        elif self.context._is_explicit_returning:
            raise exc.InvalidRequestError("Can't call inserted_primary_key when returning() is used.")
        return self.context.inserted_primary_key_rows

    @property
    def inserted_primary_key(self):
        """Return the primary key for the row just inserted.

        The return value is a :class:`_result.Row` object representing
        a named tuple of primary key values in the order in which the
        primary key columns are configured in the source
        :class:`_schema.Table`.

        .. versionchanged:: 1.4.8 - the
           :attr:`_engine.CursorResult.inserted_primary_key`
           value is now a named tuple via the :class:`_result.Row` class,
           rather than a plain tuple.

        This accessor only applies to single row :func:`_expression.insert`
        constructs which did not explicitly specify
        :meth:`_expression.Insert.returning`.    Support for multirow inserts,
        while not yet available for most backends, would be accessed using
        the :attr:`_engine.CursorResult.inserted_primary_key_rows` accessor.

        Note that primary key columns which specify a server_default clause, or
        otherwise do not qualify as "autoincrement" columns (see the notes at
        :class:`_schema.Column`), and were generated using the database-side
        default, will appear in this list as ``None`` unless the backend
        supports "returning" and the insert statement executed with the
        "implicit returning" enabled.

        Raises :class:`~sqlalchemy.exc.InvalidRequestError` if the executed
        statement is not a compiled expression construct
        or is not an insert() construct.

        """
        if self.context.executemany:
            raise exc.InvalidRequestError('This statement was an executemany call; if primary key returning is supported, please use .inserted_primary_key_rows.')
        ikp = self.inserted_primary_key_rows
        if ikp:
            return ikp[0]
        else:
            return None

    def last_updated_params(self):
        """Return the collection of updated parameters from this
        execution.

        Raises :class:`~sqlalchemy.exc.InvalidRequestError` if the executed
        statement is not a compiled expression construct
        or is not an update() construct.

        """
        if not self.context.compiled:
            raise exc.InvalidRequestError('Statement is not a compiled expression construct.')
        elif not self.context.isupdate:
            raise exc.InvalidRequestError('Statement is not an update() expression construct.')
        elif self.context.executemany:
            return self.context.compiled_parameters
        else:
            return self.context.compiled_parameters[0]

    def last_inserted_params(self):
        """Return the collection of inserted parameters from this
        execution.

        Raises :class:`~sqlalchemy.exc.InvalidRequestError` if the executed
        statement is not a compiled expression construct
        or is not an insert() construct.

        """
        if not self.context.compiled:
            raise exc.InvalidRequestError('Statement is not a compiled expression construct.')
        elif not self.context.isinsert:
            raise exc.InvalidRequestError('Statement is not an insert() expression construct.')
        elif self.context.executemany:
            return self.context.compiled_parameters
        else:
            return self.context.compiled_parameters[0]

    @property
    def returned_defaults_rows(self):
        """Return a list of rows each containing the values of default
        columns that were fetched using
        the :meth:`.ValuesBase.return_defaults` feature.

        The return value is a list of :class:`.Row` objects.

        .. versionadded:: 1.4

        """
        return self.context.returned_default_rows

    def splice_horizontally(self, other):
        """Return a new :class:`.CursorResult` that "horizontally splices"
        together the rows of this :class:`.CursorResult` with that of another
        :class:`.CursorResult`.

        .. tip::  This method is for the benefit of the SQLAlchemy ORM and is
           not intended for general use.

        "horizontally splices" means that for each row in the first and second
        result sets, a new row that concatenates the two rows together is
        produced, which then becomes the new row.  The incoming
        :class:`.CursorResult` must have the identical number of rows.  It is
        typically expected that the two result sets come from the same sort
        order as well, as the result rows are spliced together based on their
        position in the result.

        The expected use case here is so that multiple INSERT..RETURNING
        statements (which definitely need to be sorted) against different
        tables can produce a single result that looks like a JOIN of those two
        tables.

        E.g.::

            r1 = connection.execute(
                users.insert().returning(
                    users.c.user_name,
                    users.c.user_id,
                    sort_by_parameter_order=True
                ),
                user_values
            )

            r2 = connection.execute(
                addresses.insert().returning(
                    addresses.c.address_id,
                    addresses.c.address,
                    addresses.c.user_id,
                    sort_by_parameter_order=True
                ),
                address_values
            )

            rows = r1.splice_horizontally(r2).all()
            assert (
                rows ==
                [
                    ("john", 1, 1, "foo@bar.com", 1),
                    ("jack", 2, 2, "bar@bat.com", 2),
                ]
            )

        .. versionadded:: 2.0

        .. seealso::

            :meth:`.CursorResult.splice_vertically`


        """
        clone = self._generate()
        total_rows = [tuple(r1) + tuple(r2) for r1, r2 in zip(list(self._raw_row_iterator()), list(other._raw_row_iterator()))]
        clone._metadata = clone._metadata._splice_horizontally(other._metadata)
        clone.cursor_strategy = FullyBufferedCursorFetchStrategy(None, initial_buffer=total_rows)
        clone._reset_memoizations()
        return clone

    def splice_vertically(self, other):
        """Return a new :class:`.CursorResult` that "vertically splices",
        i.e. "extends", the rows of this :class:`.CursorResult` with that of
        another :class:`.CursorResult`.

        .. tip::  This method is for the benefit of the SQLAlchemy ORM and is
           not intended for general use.

        "vertically splices" means the rows of the given result are appended to
        the rows of this cursor result. The incoming :class:`.CursorResult`
        must have rows that represent the identical list of columns in the
        identical order as they are in this :class:`.CursorResult`.

        .. versionadded:: 2.0

        .. seealso::

            :meth:`.CursorResult.splice_horizontally`

        """
        clone = self._generate()
        total_rows = list(self._raw_row_iterator()) + list(other._raw_row_iterator())
        clone.cursor_strategy = FullyBufferedCursorFetchStrategy(None, initial_buffer=total_rows)
        clone._reset_memoizations()
        return clone

    def _rewind(self, rows):
        """rewind this result back to the given rowset.

        this is used internally for the case where an :class:`.Insert`
        construct combines the use of
        :meth:`.Insert.return_defaults` along with the
        "supplemental columns" feature.

        """
        if self._echo:
            self.context.connection._log_debug('CursorResult rewound %d row(s)', len(rows))
        self._metadata = cast(CursorResultMetaData, self._metadata)._remove_processors()
        self.cursor_strategy = FullyBufferedCursorFetchStrategy(None, initial_buffer=rows)
        self._reset_memoizations()
        return self

    @property
    def returned_defaults(self):
        """Return the values of default columns that were fetched using
        the :meth:`.ValuesBase.return_defaults` feature.

        The value is an instance of :class:`.Row`, or ``None``
        if :meth:`.ValuesBase.return_defaults` was not used or if the
        backend does not support RETURNING.

        .. seealso::

            :meth:`.ValuesBase.return_defaults`

        """
        if self.context.executemany:
            raise exc.InvalidRequestError('This statement was an executemany call; if return defaults is supported, please use .returned_defaults_rows.')
        rows = self.context.returned_default_rows
        if rows:
            return rows[0]
        else:
            return None

    def lastrow_has_defaults(self):
        """Return ``lastrow_has_defaults()`` from the underlying
        :class:`.ExecutionContext`.

        See :class:`.ExecutionContext` for details.

        """
        return self.context.lastrow_has_defaults()

    def postfetch_cols(self):
        """Return ``postfetch_cols()`` from the underlying
        :class:`.ExecutionContext`.

        See :class:`.ExecutionContext` for details.

        Raises :class:`~sqlalchemy.exc.InvalidRequestError` if the executed
        statement is not a compiled expression construct
        or is not an insert() or update() construct.

        """
        if not self.context.compiled:
            raise exc.InvalidRequestError('Statement is not a compiled expression construct.')
        elif not self.context.isinsert and (not self.context.isupdate):
            raise exc.InvalidRequestError('Statement is not an insert() or update() expression construct.')
        return self.context.postfetch_cols

    def prefetch_cols(self):
        """Return ``prefetch_cols()`` from the underlying
        :class:`.ExecutionContext`.

        See :class:`.ExecutionContext` for details.

        Raises :class:`~sqlalchemy.exc.InvalidRequestError` if the executed
        statement is not a compiled expression construct
        or is not an insert() or update() construct.

        """
        if not self.context.compiled:
            raise exc.InvalidRequestError('Statement is not a compiled expression construct.')
        elif not self.context.isinsert and (not self.context.isupdate):
            raise exc.InvalidRequestError('Statement is not an insert() or update() expression construct.')
        return self.context.prefetch_cols

    def supports_sane_rowcount(self):
        """Return ``supports_sane_rowcount`` from the dialect.

        See :attr:`_engine.CursorResult.rowcount` for background.

        """
        return self.dialect.supports_sane_rowcount

    def supports_sane_multi_rowcount(self):
        """Return ``supports_sane_multi_rowcount`` from the dialect.

        See :attr:`_engine.CursorResult.rowcount` for background.

        """
        return self.dialect.supports_sane_multi_rowcount

    @util.memoized_property
    def rowcount(self) -> int:
        """Return the 'rowcount' for this result.

        The primary purpose of 'rowcount' is to report the number of rows
        matched by the WHERE criterion of an UPDATE or DELETE statement
        executed once (i.e. for a single parameter set), which may then be
        compared to the number of rows expected to be updated or deleted as a
        means of asserting data integrity.

        This attribute is transferred from the ``cursor.rowcount`` attribute
        of the DBAPI before the cursor is closed, to support DBAPIs that
        don't make this value available after cursor close.   Some DBAPIs may
        offer meaningful values for other kinds of statements, such as INSERT
        and SELECT statements as well.  In order to retrieve ``cursor.rowcount``
        for these statements, set the
        :paramref:`.Connection.execution_options.preserve_rowcount`
        execution option to True, which will cause the ``cursor.rowcount``
        value to be unconditionally memoized before any results are returned
        or the cursor is closed, regardless of statement type.

        For cases where the DBAPI does not support rowcount for a particular
        kind of statement and/or execution, the returned value will be ``-1``,
        which is delivered directly from the DBAPI and is part of :pep:`249`.
        All DBAPIs should support rowcount for single-parameter-set
        UPDATE and DELETE statements, however.

        .. note::

           Notes regarding :attr:`_engine.CursorResult.rowcount`:


           * This attribute returns the number of rows *matched*,
             which is not necessarily the same as the number of rows
             that were actually *modified*. For example, an UPDATE statement
             may have no net change on a given row if the SET values
             given are the same as those present in the row already.
             Such a row would be matched but not modified.
             On backends that feature both styles, such as MySQL,
             rowcount is configured to return the match
             count in all cases.

           * :attr:`_engine.CursorResult.rowcount` in the default case is
             *only* useful in conjunction with an UPDATE or DELETE statement,
             and only with a single set of parameters. For other kinds of
             statements, SQLAlchemy will not attempt to pre-memoize the value
             unless the
             :paramref:`.Connection.execution_options.preserve_rowcount`
             execution option is used.  Note that contrary to :pep:`249`, many
             DBAPIs do not support rowcount values for statements that are not
             UPDATE or DELETE, particularly when rows are being returned which
             are not fully pre-buffered.   DBAPIs that dont support rowcount
             for a particular kind of statement should return the value ``-1``
             for such statements.

           * :attr:`_engine.CursorResult.rowcount` may not be meaningful
             when executing a single statement with multiple parameter sets
             (i.e. an :term:`executemany`). Most DBAPIs do not sum "rowcount"
             values across multiple parameter sets and will return ``-1``
             when accessed.

           * SQLAlchemy's :ref:`engine_insertmanyvalues` feature does support
             a correct population of :attr:`_engine.CursorResult.rowcount`
             when the :paramref:`.Connection.execution_options.preserve_rowcount`
             execution option is set to True.

           * Statements that use RETURNING may not support rowcount, returning
             a ``-1`` value instead.

        .. seealso::

            :ref:`tutorial_update_delete_rowcount` - in the :ref:`unified_tutorial`

            :paramref:`.Connection.execution_options.preserve_rowcount`

        """
        try:
            return self.context.rowcount
        except BaseException as e:
            self.cursor_strategy.handle_exception(self, self.cursor, e)
            raise

    @property
    def lastrowid(self):
        """Return the 'lastrowid' accessor on the DBAPI cursor.

        This is a DBAPI specific method and is only functional
        for those backends which support it, for statements
        where it is appropriate.  It's behavior is not
        consistent across backends.

        Usage of this method is normally unnecessary when
        using insert() expression constructs; the
        :attr:`~CursorResult.inserted_primary_key` attribute provides a
        tuple of primary key values for a newly inserted row,
        regardless of database backend.

        """
        try:
            return self.context.get_lastrowid()
        except BaseException as e:
            self.cursor_strategy.handle_exception(self, self.cursor, e)

    @property
    def returns_rows(self):
        """True if this :class:`_engine.CursorResult` returns zero or more
        rows.

        I.e. if it is legal to call the methods
        :meth:`_engine.CursorResult.fetchone`,
        :meth:`_engine.CursorResult.fetchmany`
        :meth:`_engine.CursorResult.fetchall`.

        Overall, the value of :attr:`_engine.CursorResult.returns_rows` should
        always be synonymous with whether or not the DBAPI cursor had a
        ``.description`` attribute, indicating the presence of result columns,
        noting that a cursor that returns zero rows still has a
        ``.description`` if a row-returning statement was emitted.

        This attribute should be True for all results that are against
        SELECT statements, as well as for DML statements INSERT/UPDATE/DELETE
        that use RETURNING.   For INSERT/UPDATE/DELETE statements that were
        not using RETURNING, the value will usually be False, however
        there are some dialect-specific exceptions to this, such as when
        using the MSSQL / pyodbc dialect a SELECT is emitted inline in
        order to retrieve an inserted primary key value.


        """
        return self._metadata.returns_rows

    @property
    def is_insert(self):
        """True if this :class:`_engine.CursorResult` is the result
        of a executing an expression language compiled
        :func:`_expression.insert` construct.

        When True, this implies that the
        :attr:`inserted_primary_key` attribute is accessible,
        assuming the statement did not include
        a user defined "returning" construct.

        """
        return self.context.isinsert

    def _fetchiter_impl(self):
        fetchone = self.cursor_strategy.fetchone
        while True:
            row = fetchone(self, self.cursor)
            if row is None:
                break
            yield row

    def _fetchone_impl(self, hard_close=False):
        return self.cursor_strategy.fetchone(self, self.cursor, hard_close)

    def _fetchall_impl(self):
        return self.cursor_strategy.fetchall(self, self.cursor)

    def _fetchmany_impl(self, size=None):
        return self.cursor_strategy.fetchmany(self, self.cursor, size)

    def _raw_row_iterator(self):
        return self._fetchiter_impl()

    def merge(self, *others: Result[Any]) -> MergedResult[Any]:
        merged_result = super().merge(*others)
        if self.context._has_rowcount:
            merged_result.rowcount = sum((cast('CursorResult[Any]', result).rowcount for result in (self,) + others))
        return merged_result

    def close(self) -> Any:
        """Close this :class:`_engine.CursorResult`.

        This closes out the underlying DBAPI cursor corresponding to the
        statement execution, if one is still present.  Note that the DBAPI
        cursor is automatically released when the :class:`_engine.CursorResult`
        exhausts all available rows.  :meth:`_engine.CursorResult.close` is
        generally an optional method except in the case when discarding a
        :class:`_engine.CursorResult` that still has additional rows pending
        for fetch.

        After this method is called, it is no longer valid to call upon
        the fetch methods, which will raise a :class:`.ResourceClosedError`
        on subsequent use.

        .. seealso::

            :ref:`connections_toplevel`

        """
        self._soft_close(hard=True)

    @_generative
    def yield_per(self, num: int) -> Self:
        self._yield_per = num
        self.cursor_strategy.yield_per(self, self.cursor, num)
        return self