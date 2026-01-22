from __future__ import annotations
import functools
import operator
import random
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import MutableSequence
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import weakref
from . import characteristics
from . import cursor as _cursor
from . import interfaces
from .base import Connection
from .interfaces import CacheStats
from .interfaces import DBAPICursor
from .interfaces import Dialect
from .interfaces import ExecuteStyle
from .interfaces import ExecutionContext
from .reflection import ObjectKind
from .reflection import ObjectScope
from .. import event
from .. import exc
from .. import pool
from .. import util
from ..sql import compiler
from ..sql import dml
from ..sql import expression
from ..sql import type_api
from ..sql._typing import is_tuple_type
from ..sql.base import _NoArg
from ..sql.compiler import DDLCompiler
from ..sql.compiler import InsertmanyvaluesSentinelOpts
from ..sql.compiler import SQLCompiler
from ..sql.elements import quoted_name
from ..util.typing import Final
from ..util.typing import Literal
class DefaultDialect(Dialect):
    """Default implementation of Dialect"""
    statement_compiler = compiler.SQLCompiler
    ddl_compiler = compiler.DDLCompiler
    type_compiler_cls = compiler.GenericTypeCompiler
    preparer = compiler.IdentifierPreparer
    supports_alter = True
    supports_comments = False
    supports_constraint_comments = False
    inline_comments = False
    supports_statement_cache = True
    div_is_floordiv = True
    bind_typing = interfaces.BindTyping.NONE
    include_set_input_sizes: Optional[Set[Any]] = None
    exclude_set_input_sizes: Optional[Set[Any]] = None
    default_sequence_base = 1
    execute_sequence_format = tuple
    supports_schemas = True
    supports_views = True
    supports_sequences = False
    sequences_optional = False
    preexecute_autoincrement_sequences = False
    supports_identity_columns = False
    postfetch_lastrowid = True
    favor_returning_over_lastrowid = False
    insert_null_pk_still_autoincrements = False
    update_returning = False
    delete_returning = False
    update_returning_multifrom = False
    delete_returning_multifrom = False
    insert_returning = False
    cte_follows_insert = False
    supports_native_enum = False
    supports_native_boolean = False
    supports_native_uuid = False
    returns_native_bytes = False
    non_native_boolean_check_constraint = True
    supports_simple_order_by_label = True
    tuple_in_values = False
    connection_characteristics = util.immutabledict({'isolation_level': characteristics.IsolationLevelCharacteristic()})
    engine_config_types: Mapping[str, Any] = util.immutabledict({'pool_timeout': util.asint, 'echo': util.bool_or_str('debug'), 'echo_pool': util.bool_or_str('debug'), 'pool_recycle': util.asint, 'pool_size': util.asint, 'max_overflow': util.asint, 'future': util.asbool})
    supports_native_decimal = False
    name = 'default'
    max_identifier_length = 9999
    _user_defined_max_identifier_length: Optional[int] = None
    isolation_level: Optional[str] = None
    max_index_name_length: Optional[int] = None
    max_constraint_name_length: Optional[int] = None
    supports_sane_rowcount = True
    supports_sane_multi_rowcount = True
    colspecs: MutableMapping[Type[TypeEngine[Any]], Type[TypeEngine[Any]]] = {}
    default_paramstyle = 'named'
    supports_default_values = False
    'dialect supports INSERT... DEFAULT VALUES syntax'
    supports_default_metavalue = False
    'dialect supports INSERT... VALUES (DEFAULT) syntax'
    default_metavalue_token = 'DEFAULT'
    'for INSERT... VALUES (DEFAULT) syntax, the token to put in the\n    parenthesis.'
    supports_empty_insert = True
    'dialect supports INSERT () VALUES ()'
    supports_multivalues_insert = False
    use_insertmanyvalues: bool = False
    use_insertmanyvalues_wo_returning: bool = False
    insertmanyvalues_implicit_sentinel: InsertmanyvaluesSentinelOpts = InsertmanyvaluesSentinelOpts.NOT_SUPPORTED
    insertmanyvalues_page_size: int = 1000
    insertmanyvalues_max_parameters = 32700
    supports_is_distinct_from = True
    supports_server_side_cursors = False
    server_side_cursors = False
    supports_for_update_of = False
    server_version_info = None
    default_schema_name: Optional[str] = None
    requires_name_normalize = False
    is_async = False
    has_terminate = False
    _legacy_binary_type_literal_encoding = 'utf-8'

    @util.deprecated_params(empty_in_strategy=('1.4', 'The :paramref:`_sa.create_engine.empty_in_strategy` keyword is deprecated, and no longer has any effect.  All IN expressions are now rendered using the "expanding parameter" strategy which renders a set of boundexpressions, or an "empty set" SELECT, at statement executiontime.'), server_side_cursors=('1.4', 'The :paramref:`_sa.create_engine.server_side_cursors` parameter is deprecated and will be removed in a future release.  Please use the :paramref:`_engine.Connection.execution_options.stream_results` parameter.'))
    def __init__(self, paramstyle: Optional[_ParamStyle]=None, isolation_level: Optional[IsolationLevel]=None, dbapi: Optional[ModuleType]=None, implicit_returning: Literal[True]=True, supports_native_boolean: Optional[bool]=None, max_identifier_length: Optional[int]=None, label_length: Optional[int]=None, insertmanyvalues_page_size: Union[_NoArg, int]=_NoArg.NO_ARG, use_insertmanyvalues: Optional[bool]=None, compiler_linting: Linting=int(compiler.NO_LINTING), server_side_cursors: bool=False, **kwargs: Any):
        if server_side_cursors:
            if not self.supports_server_side_cursors:
                raise exc.ArgumentError('Dialect %s does not support server side cursors' % self)
            else:
                self.server_side_cursors = True
        if getattr(self, 'use_setinputsizes', False):
            util.warn_deprecated('The dialect-level use_setinputsizes attribute is deprecated.  Please use bind_typing = BindTyping.SETINPUTSIZES', '2.0')
            self.bind_typing = interfaces.BindTyping.SETINPUTSIZES
        self.positional = False
        self._ischema = None
        self.dbapi = dbapi
        if paramstyle is not None:
            self.paramstyle = paramstyle
        elif self.dbapi is not None:
            self.paramstyle = self.dbapi.paramstyle
        else:
            self.paramstyle = self.default_paramstyle
        self.positional = self.paramstyle in ('qmark', 'format', 'numeric', 'numeric_dollar')
        self.identifier_preparer = self.preparer(self)
        self._on_connect_isolation_level = isolation_level
        legacy_tt_callable = getattr(self, 'type_compiler', None)
        if legacy_tt_callable is not None:
            tt_callable = cast(Type[compiler.GenericTypeCompiler], self.type_compiler)
        else:
            tt_callable = self.type_compiler_cls
        self.type_compiler_instance = self.type_compiler = tt_callable(self)
        if supports_native_boolean is not None:
            self.supports_native_boolean = supports_native_boolean
        self._user_defined_max_identifier_length = max_identifier_length
        if self._user_defined_max_identifier_length:
            self.max_identifier_length = self._user_defined_max_identifier_length
        self.label_length = label_length
        self.compiler_linting = compiler_linting
        if use_insertmanyvalues is not None:
            self.use_insertmanyvalues = use_insertmanyvalues
        if insertmanyvalues_page_size is not _NoArg.NO_ARG:
            self.insertmanyvalues_page_size = insertmanyvalues_page_size

    @property
    @util.deprecated('2.0', 'full_returning is deprecated, please use insert_returning, update_returning, delete_returning')
    def full_returning(self):
        return self.insert_returning and self.update_returning and self.delete_returning

    @util.memoized_property
    def insert_executemany_returning(self):
        """Default implementation for insert_executemany_returning, if not
        otherwise overridden by the specific dialect.

        The default dialect determines "insert_executemany_returning" is
        available if the dialect in use has opted into using the
        "use_insertmanyvalues" feature. If they haven't opted into that, then
        this attribute is False, unless the dialect in question overrides this
        and provides some other implementation (such as the Oracle dialect).

        """
        return self.insert_returning and self.use_insertmanyvalues

    @util.memoized_property
    def insert_executemany_returning_sort_by_parameter_order(self):
        """Default implementation for
        insert_executemany_returning_deterministic_order, if not otherwise
        overridden by the specific dialect.

        The default dialect determines "insert_executemany_returning" can have
        deterministic order only if the dialect in use has opted into using the
        "use_insertmanyvalues" feature, which implements deterministic ordering
        using client side sentinel columns only by default.  The
        "insertmanyvalues" feature also features alternate forms that can
        use server-generated PK values as "sentinels", but those are only
        used if the :attr:`.Dialect.insertmanyvalues_implicit_sentinel`
        bitflag enables those alternate SQL forms, which are disabled
        by default.

        If the dialect in use hasn't opted into that, then this attribute is
        False, unless the dialect in question overrides this and provides some
        other implementation (such as the Oracle dialect).

        """
        return self.insert_returning and self.use_insertmanyvalues
    update_executemany_returning = False
    delete_executemany_returning = False

    @util.memoized_property
    def loaded_dbapi(self) -> ModuleType:
        if self.dbapi is None:
            raise exc.InvalidRequestError(f'Dialect {self} does not have a Python DBAPI established and cannot be used for actual database interaction')
        return self.dbapi

    @util.memoized_property
    def _bind_typing_render_casts(self):
        return self.bind_typing is interfaces.BindTyping.RENDER_CASTS

    def _ensure_has_table_connection(self, arg):
        if not isinstance(arg, Connection):
            raise exc.ArgumentError('The argument passed to Dialect.has_table() should be a %s, got %s. Additionally, the Dialect.has_table() method is for internal dialect use only; please use ``inspect(some_engine).has_table(<tablename>>)`` for public API use.' % (Connection, type(arg)))

    @util.memoized_property
    def _supports_statement_cache(self):
        ssc = self.__class__.__dict__.get('supports_statement_cache', None)
        if ssc is None:
            util.warn("Dialect %s:%s will not make use of SQL compilation caching as it does not set the 'supports_statement_cache' attribute to ``True``.  This can have significant performance implications including some performance degradations in comparison to prior SQLAlchemy versions.  Dialect maintainers should seek to set this attribute to True after appropriate development and testing for SQLAlchemy 1.4 caching support.   Alternatively, this attribute may be set to False which will disable this warning." % (self.name, self.driver), code='cprf')
        return bool(ssc)

    @util.memoized_property
    def _type_memos(self):
        return weakref.WeakKeyDictionary()

    @property
    def dialect_description(self):
        return self.name + '+' + self.driver

    @property
    def supports_sane_rowcount_returning(self):
        """True if this dialect supports sane rowcount even if RETURNING is
        in use.

        For dialects that don't support RETURNING, this is synonymous with
        ``supports_sane_rowcount``.

        """
        return self.supports_sane_rowcount

    @classmethod
    def get_pool_class(cls, url: URL) -> Type[Pool]:
        return getattr(cls, 'poolclass', pool.QueuePool)

    def get_dialect_pool_class(self, url: URL) -> Type[Pool]:
        return self.get_pool_class(url)

    @classmethod
    def load_provisioning(cls):
        package = '.'.join(cls.__module__.split('.')[0:-1])
        try:
            __import__(package + '.provision')
        except ImportError:
            pass

    def _builtin_onconnect(self) -> Optional[_ListenerFnType]:
        if self._on_connect_isolation_level is not None:

            def builtin_connect(dbapi_conn, conn_rec):
                self._assert_and_set_isolation_level(dbapi_conn, self._on_connect_isolation_level)
            return builtin_connect
        else:
            return None

    def initialize(self, connection):
        try:
            self.server_version_info = self._get_server_version_info(connection)
        except NotImplementedError:
            self.server_version_info = None
        try:
            self.default_schema_name = self._get_default_schema_name(connection)
        except NotImplementedError:
            self.default_schema_name = None
        try:
            self.default_isolation_level = self.get_default_isolation_level(connection.connection.dbapi_connection)
        except NotImplementedError:
            self.default_isolation_level = None
        if not self._user_defined_max_identifier_length:
            max_ident_length = self._check_max_identifier_length(connection)
            if max_ident_length:
                self.max_identifier_length = max_ident_length
        if self.label_length and self.label_length > self.max_identifier_length:
            raise exc.ArgumentError("Label length of %d is greater than this dialect's maximum identifier length of %d" % (self.label_length, self.max_identifier_length))

    def on_connect(self):
        return None

    def _check_max_identifier_length(self, connection):
        """Perform a connection / server version specific check to determine
        the max_identifier_length.

        If the dialect's class level max_identifier_length should be used,
        can return None.

        .. versionadded:: 1.3.9

        """
        return None

    def get_default_isolation_level(self, dbapi_conn):
        """Given a DBAPI connection, return its isolation level, or
        a default isolation level if one cannot be retrieved.

        May be overridden by subclasses in order to provide a
        "fallback" isolation level for databases that cannot reliably
        retrieve the actual isolation level.

        By default, calls the :meth:`_engine.Interfaces.get_isolation_level`
        method, propagating any exceptions raised.

        .. versionadded:: 1.3.22

        """
        return self.get_isolation_level(dbapi_conn)

    def type_descriptor(self, typeobj):
        """Provide a database-specific :class:`.TypeEngine` object, given
        the generic object which comes from the types module.

        This method looks for a dictionary called
        ``colspecs`` as a class or instance-level variable,
        and passes on to :func:`_types.adapt_type`.

        """
        return type_api.adapt_type(typeobj, self.colspecs)

    def has_index(self, connection, table_name, index_name, schema=None, **kw):
        if not self.has_table(connection, table_name, schema=schema, **kw):
            return False
        for idx in self.get_indexes(connection, table_name, schema=schema, **kw):
            if idx['name'] == index_name:
                return True
        else:
            return False

    def has_schema(self, connection: Connection, schema_name: str, **kw: Any) -> bool:
        return schema_name in self.get_schema_names(connection, **kw)

    def validate_identifier(self, ident):
        if len(ident) > self.max_identifier_length:
            raise exc.IdentifierError("Identifier '%s' exceeds maximum length of %d characters" % (ident, self.max_identifier_length))

    def connect(self, *cargs, **cparams):
        return self.loaded_dbapi.connect(*cargs, **cparams)

    def create_connect_args(self, url):
        opts = url.translate_connect_args()
        opts.update(url.query)
        return ([], opts)

    def set_engine_execution_options(self, engine: Engine, opts: Mapping[str, Any]) -> None:
        supported_names = set(self.connection_characteristics).intersection(opts)
        if supported_names:
            characteristics: Mapping[str, Any] = util.immutabledict(((name, opts[name]) for name in supported_names))

            @event.listens_for(engine, 'engine_connect')
            def set_connection_characteristics(connection):
                self._set_connection_characteristics(connection, characteristics)

    def set_connection_execution_options(self, connection: Connection, opts: Mapping[str, Any]) -> None:
        supported_names = set(self.connection_characteristics).intersection(opts)
        if supported_names:
            characteristics: Mapping[str, Any] = util.immutabledict(((name, opts[name]) for name in supported_names))
            self._set_connection_characteristics(connection, characteristics)

    def _set_connection_characteristics(self, connection, characteristics):
        characteristic_values = [(name, self.connection_characteristics[name], value) for name, value in characteristics.items()]
        if connection.in_transaction():
            trans_objs = [(name, obj) for name, obj, value in characteristic_values if obj.transactional]
            if trans_objs:
                raise exc.InvalidRequestError('This connection has already initialized a SQLAlchemy Transaction() object via begin() or autobegin; %s may not be altered unless rollback() or commit() is called first.' % ', '.join((name for name, obj in trans_objs)))
        dbapi_connection = connection.connection.dbapi_connection
        for name, characteristic, value in characteristic_values:
            characteristic.set_characteristic(self, dbapi_connection, value)
        connection.connection._connection_record.finalize_callback.append(functools.partial(self._reset_characteristics, characteristics))

    def _reset_characteristics(self, characteristics, dbapi_connection):
        for characteristic_name in characteristics:
            characteristic = self.connection_characteristics[characteristic_name]
            characteristic.reset_characteristic(self, dbapi_connection)

    def do_begin(self, dbapi_connection):
        pass

    def do_rollback(self, dbapi_connection):
        dbapi_connection.rollback()

    def do_commit(self, dbapi_connection):
        dbapi_connection.commit()

    def do_terminate(self, dbapi_connection):
        self.do_close(dbapi_connection)

    def do_close(self, dbapi_connection):
        dbapi_connection.close()

    @util.memoized_property
    def _dialect_specific_select_one(self):
        return str(expression.select(1).compile(dialect=self))

    def _do_ping_w_event(self, dbapi_connection: DBAPIConnection) -> bool:
        try:
            return self.do_ping(dbapi_connection)
        except self.loaded_dbapi.Error as err:
            is_disconnect = self.is_disconnect(err, dbapi_connection, None)
            if self._has_events:
                try:
                    Connection._handle_dbapi_exception_noconnection(err, self, is_disconnect=is_disconnect, invalidate_pool_on_disconnect=False, is_pre_ping=True)
                except exc.StatementError as new_err:
                    is_disconnect = new_err.connection_invalidated
            if is_disconnect:
                return False
            else:
                raise

    def do_ping(self, dbapi_connection: DBAPIConnection) -> bool:
        cursor = None
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(self._dialect_specific_select_one)
        finally:
            cursor.close()
        return True

    def create_xid(self):
        """Create a random two-phase transaction ID.

        This id will be passed to do_begin_twophase(), do_rollback_twophase(),
        do_commit_twophase().  Its format is unspecified.
        """
        return '_sa_%032x' % random.randint(0, 2 ** 128)

    def do_savepoint(self, connection, name):
        connection.execute(expression.SavepointClause(name))

    def do_rollback_to_savepoint(self, connection, name):
        connection.execute(expression.RollbackToSavepointClause(name))

    def do_release_savepoint(self, connection, name):
        connection.execute(expression.ReleaseSavepointClause(name))

    def _deliver_insertmanyvalues_batches(self, cursor, statement, parameters, generic_setinputsizes, context):
        context = cast(DefaultExecutionContext, context)
        compiled = cast(SQLCompiler, context.compiled)
        _composite_sentinel_proc: Sequence[Optional[_ResultProcessorType[Any]]] = ()
        _scalar_sentinel_proc: Optional[_ResultProcessorType[Any]] = None
        _sentinel_proc_initialized: bool = False
        compiled_parameters = context.compiled_parameters
        imv = compiled._insertmanyvalues
        assert imv is not None
        is_returning: Final[bool] = bool(compiled.effective_returning)
        batch_size = context.execution_options.get('insertmanyvalues_page_size', self.insertmanyvalues_page_size)
        if compiled.schema_translate_map:
            schema_translate_map = context.execution_options.get('schema_translate_map', {})
        else:
            schema_translate_map = None
        if is_returning:
            result: Optional[List[Any]] = []
            context._insertmanyvalues_rows = result
            sort_by_parameter_order = imv.sort_by_parameter_order
        else:
            sort_by_parameter_order = False
            result = None
        for imv_batch in compiled._deliver_insertmanyvalues_batches(statement, parameters, compiled_parameters, generic_setinputsizes, batch_size, sort_by_parameter_order, schema_translate_map):
            yield imv_batch
            if is_returning:
                rows = context.fetchall_for_returning(cursor)
                assert result is not None
                if imv.num_sentinel_columns and (not imv_batch.is_downgraded):
                    composite_sentinel = imv.num_sentinel_columns > 1
                    if imv.implicit_sentinel:
                        assert not composite_sentinel
                        result.extend(sorted(rows, key=operator.itemgetter(-1)))
                        continue
                    assert imv.sentinel_param_keys
                    assert imv.sentinel_columns
                    _nsc = imv.num_sentinel_columns
                    if not _sentinel_proc_initialized:
                        if composite_sentinel:
                            _composite_sentinel_proc = [col.type._cached_result_processor(self, cursor_desc[1]) for col, cursor_desc in zip(imv.sentinel_columns, cursor.description[-_nsc:])]
                        else:
                            _scalar_sentinel_proc = imv.sentinel_columns[0].type._cached_result_processor(self, cursor.description[-1][1])
                        _sentinel_proc_initialized = True
                    rows_by_sentinel: Union[Dict[Tuple[Any, ...], Any], Dict[Any, Any]]
                    if composite_sentinel:
                        rows_by_sentinel = {tuple((proc(val) if proc else val for val, proc in zip(row[-_nsc:], _composite_sentinel_proc))): row for row in rows}
                    elif _scalar_sentinel_proc:
                        rows_by_sentinel = {_scalar_sentinel_proc(row[-1]): row for row in rows}
                    else:
                        rows_by_sentinel = {row[-1]: row for row in rows}
                    if len(rows_by_sentinel) != len(imv_batch.batch):
                        raise exc.InvalidRequestError(f'Sentinel-keyed result set did not produce correct number of rows {len(imv_batch.batch)}; produced {len(rows_by_sentinel)}.  Please ensure the sentinel column is fully unique and populated in all cases.')
                    try:
                        ordered_rows = [rows_by_sentinel[sentinel_keys] for sentinel_keys in imv_batch.sentinel_values]
                    except KeyError as ke:
                        raise exc.InvalidRequestError(f"Can't match sentinel values in result set to parameter sets; key {ke.args[0]!r} was not found. There may be a mismatch between the datatype passed to the DBAPI driver vs. that which it returns in a result row.  Ensure the given Python value matches the expected result type *exactly*, taking care to not rely upon implicit conversions which may occur such as when using strings in place of UUID or integer values, etc. ") from ke
                    result.extend(ordered_rows)
                else:
                    result.extend(rows)

    def do_executemany(self, cursor, statement, parameters, context=None):
        cursor.executemany(statement, parameters)

    def do_execute(self, cursor, statement, parameters, context=None):
        cursor.execute(statement, parameters)

    def do_execute_no_params(self, cursor, statement, context=None):
        cursor.execute(statement)

    def is_disconnect(self, e, connection, cursor):
        return False

    @util.memoized_instancemethod
    def _gen_allowed_isolation_levels(self, dbapi_conn):
        try:
            raw_levels = list(self.get_isolation_level_values(dbapi_conn))
        except NotImplementedError:
            return None
        else:
            normalized_levels = [level.replace('_', ' ').upper() for level in raw_levels]
            if raw_levels != normalized_levels:
                raise ValueError(f'Dialect {self.name!r} get_isolation_level_values() method should return names as UPPERCASE using spaces, not underscores; got {sorted(set(raw_levels).difference(normalized_levels))}')
            return tuple(normalized_levels)

    def _assert_and_set_isolation_level(self, dbapi_conn, level):
        level = level.replace('_', ' ').upper()
        _allowed_isolation_levels = self._gen_allowed_isolation_levels(dbapi_conn)
        if _allowed_isolation_levels and level not in _allowed_isolation_levels:
            raise exc.ArgumentError(f'Invalid value {level!r} for isolation_level. Valid isolation levels for {self.name!r} are {', '.join(_allowed_isolation_levels)}')
        self.set_isolation_level(dbapi_conn, level)

    def reset_isolation_level(self, dbapi_conn):
        if self._on_connect_isolation_level is not None:
            assert self._on_connect_isolation_level == 'AUTOCOMMIT' or self._on_connect_isolation_level == self.default_isolation_level
            self._assert_and_set_isolation_level(dbapi_conn, self._on_connect_isolation_level)
        else:
            assert self.default_isolation_level is not None
            self._assert_and_set_isolation_level(dbapi_conn, self.default_isolation_level)

    def normalize_name(self, name):
        if name is None:
            return None
        name_lower = name.lower()
        name_upper = name.upper()
        if name_upper == name_lower:
            return name
        elif name_upper == name and (not self.identifier_preparer._requires_quotes(name_lower)):
            return name_lower
        elif name_lower == name:
            return quoted_name(name, quote=True)
        else:
            return name

    def denormalize_name(self, name):
        if name is None:
            return None
        name_lower = name.lower()
        name_upper = name.upper()
        if name_upper == name_lower:
            return name
        elif name_lower == name and (not self.identifier_preparer._requires_quotes(name_lower)):
            name = name_upper
        return name

    def get_driver_connection(self, connection):
        return connection

    def _overrides_default(self, method):
        return getattr(type(self), method).__code__ is not getattr(DefaultDialect, method).__code__

    def _default_multi_reflect(self, single_tbl_method, connection, kind, schema, filter_names, scope, **kw):
        names_fns = []
        temp_names_fns = []
        if ObjectKind.TABLE in kind:
            names_fns.append(self.get_table_names)
            temp_names_fns.append(self.get_temp_table_names)
        if ObjectKind.VIEW in kind:
            names_fns.append(self.get_view_names)
            temp_names_fns.append(self.get_temp_view_names)
        if ObjectKind.MATERIALIZED_VIEW in kind:
            names_fns.append(self.get_materialized_view_names)
        unreflectable = kw.pop('unreflectable', {})
        if filter_names and scope is ObjectScope.ANY and (kind is ObjectKind.ANY):
            names = filter_names
        else:
            names = []
            name_kw = {'schema': schema, **kw}
            fns = []
            if ObjectScope.DEFAULT in scope:
                fns.extend(names_fns)
            if ObjectScope.TEMPORARY in scope:
                fns.extend(temp_names_fns)
            for fn in fns:
                try:
                    names.extend(fn(connection, **name_kw))
                except NotImplementedError:
                    pass
        if filter_names:
            filter_names = set(filter_names)
        for table in names:
            if not filter_names or table in filter_names:
                key = (schema, table)
                try:
                    yield (key, single_tbl_method(connection, table, schema=schema, **kw))
                except exc.UnreflectableTableError as err:
                    if key not in unreflectable:
                        unreflectable[key] = err
                except exc.NoSuchTableError:
                    pass

    def get_multi_table_options(self, connection, **kw):
        return self._default_multi_reflect(self.get_table_options, connection, **kw)

    def get_multi_columns(self, connection, **kw):
        return self._default_multi_reflect(self.get_columns, connection, **kw)

    def get_multi_pk_constraint(self, connection, **kw):
        return self._default_multi_reflect(self.get_pk_constraint, connection, **kw)

    def get_multi_foreign_keys(self, connection, **kw):
        return self._default_multi_reflect(self.get_foreign_keys, connection, **kw)

    def get_multi_indexes(self, connection, **kw):
        return self._default_multi_reflect(self.get_indexes, connection, **kw)

    def get_multi_unique_constraints(self, connection, **kw):
        return self._default_multi_reflect(self.get_unique_constraints, connection, **kw)

    def get_multi_check_constraints(self, connection, **kw):
        return self._default_multi_reflect(self.get_check_constraints, connection, **kw)

    def get_multi_table_comment(self, connection, **kw):
        return self._default_multi_reflect(self.get_table_comment, connection, **kw)