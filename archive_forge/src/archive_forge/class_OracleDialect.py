from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from functools import wraps
import re
from . import dictionary
from .types import _OracleBoolean
from .types import _OracleDate
from .types import BFILE
from .types import BINARY_DOUBLE
from .types import BINARY_FLOAT
from .types import DATE
from .types import FLOAT
from .types import INTERVAL
from .types import LONG
from .types import NCLOB
from .types import NUMBER
from .types import NVARCHAR2  # noqa
from .types import OracleRaw  # noqa
from .types import RAW
from .types import ROWID  # noqa
from .types import TIMESTAMP
from .types import VARCHAR2  # noqa
from ... import Computed
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import util
from ...engine import default
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import and_
from ...sql import bindparam
from ...sql import compiler
from ...sql import expression
from ...sql import func
from ...sql import null
from ...sql import or_
from ...sql import select
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql import visitors
from ...sql.visitors import InternalTraversal
from ...types import BLOB
from ...types import CHAR
from ...types import CLOB
from ...types import DOUBLE_PRECISION
from ...types import INTEGER
from ...types import NCHAR
from ...types import NVARCHAR
from ...types import REAL
from ...types import VARCHAR
class OracleDialect(default.DefaultDialect):
    name = 'oracle'
    supports_statement_cache = True
    supports_alter = True
    max_identifier_length = 128
    _supports_offset_fetch = True
    insert_returning = True
    update_returning = True
    delete_returning = True
    div_is_floordiv = False
    supports_simple_order_by_label = False
    cte_follows_insert = True
    returns_native_bytes = True
    supports_sequences = True
    sequences_optional = False
    postfetch_lastrowid = False
    default_paramstyle = 'named'
    colspecs = colspecs
    ischema_names = ischema_names
    requires_name_normalize = True
    supports_comments = True
    supports_default_values = False
    supports_default_metavalue = True
    supports_empty_insert = False
    supports_identity_columns = True
    statement_compiler = OracleCompiler
    ddl_compiler = OracleDDLCompiler
    type_compiler_cls = OracleTypeCompiler
    preparer = OracleIdentifierPreparer
    execution_ctx_cls = OracleExecutionContext
    reflection_options = ('oracle_resolve_synonyms',)
    _use_nchar_for_unicode = False
    construct_arguments = [(sa_schema.Table, {'resolve_synonyms': False, 'on_commit': None, 'compress': False}), (sa_schema.Index, {'bitmap': False, 'compress': False})]

    @util.deprecated_params(use_binds_for_limits=('1.4', "The ``use_binds_for_limits`` Oracle dialect parameter is deprecated. The dialect now renders LIMIT /OFFSET integers inline in all cases using a post-compilation hook, so that the value is still represented by a 'bound parameter' on the Core Expression side."))
    def __init__(self, use_ansi=True, optimize_limits=False, use_binds_for_limits=None, use_nchar_for_unicode=False, exclude_tablespaces=('SYSTEM', 'SYSAUX'), enable_offset_fetch=True, **kwargs):
        default.DefaultDialect.__init__(self, **kwargs)
        self._use_nchar_for_unicode = use_nchar_for_unicode
        self.use_ansi = use_ansi
        self.optimize_limits = optimize_limits
        self.exclude_tablespaces = exclude_tablespaces
        self.enable_offset_fetch = self._supports_offset_fetch = enable_offset_fetch

    def initialize(self, connection):
        super().initialize(connection)
        if self._is_oracle_8:
            self.colspecs = self.colspecs.copy()
            self.colspecs.pop(sqltypes.Interval)
            self.use_ansi = False
        self.supports_identity_columns = self.server_version_info >= (12,)
        self._supports_offset_fetch = self.enable_offset_fetch and self.server_version_info >= (12,)

    def _get_effective_compat_server_version_info(self, connection):
        if self.server_version_info < (12, 2):
            return self.server_version_info
        try:
            compat = connection.exec_driver_sql("SELECT value FROM v$parameter WHERE name = 'compatible'").scalar()
        except exc.DBAPIError:
            compat = None
        if compat:
            try:
                return tuple((int(x) for x in compat.split('.')))
            except:
                return self.server_version_info
        else:
            return self.server_version_info

    @property
    def _is_oracle_8(self):
        return self.server_version_info and self.server_version_info < (9,)

    @property
    def _supports_table_compression(self):
        return self.server_version_info and self.server_version_info >= (10, 1)

    @property
    def _supports_table_compress_for(self):
        return self.server_version_info and self.server_version_info >= (11,)

    @property
    def _supports_char_length(self):
        return not self._is_oracle_8

    @property
    def _supports_update_returning_computed_cols(self):
        return self.server_version_info and self.server_version_info >= (18,)

    @property
    def _supports_except_all(self):
        return self.server_version_info and self.server_version_info >= (21,)

    def do_release_savepoint(self, connection, name):
        pass

    def _check_max_identifier_length(self, connection):
        if self._get_effective_compat_server_version_info(connection) < (12, 2):
            return 30
        else:
            return None

    def get_isolation_level_values(self, dbapi_connection):
        return ['READ COMMITTED', 'SERIALIZABLE']

    def get_default_isolation_level(self, dbapi_conn):
        try:
            return self.get_isolation_level(dbapi_conn)
        except NotImplementedError:
            raise
        except:
            return 'READ COMMITTED'

    def _execute_reflection(self, connection, query, dblink, returns_long, params=None):
        if dblink and (not dblink.startswith('@')):
            dblink = f'@{dblink}'
        execution_options = {'_oracle_dblink': dblink or '', 'schema_translate_map': None}
        if dblink and returns_long:

            def visit_bindparam(bindparam):
                bindparam.literal_execute = True
            query = visitors.cloned_traverse(query, {}, {'bindparam': visit_bindparam})
        return connection.execute(query, params, execution_options=execution_options)

    @util.memoized_property
    def _has_table_query(self):
        tables = select(dictionary.all_tables.c.table_name, dictionary.all_tables.c.owner).union_all(select(dictionary.all_views.c.view_name.label('table_name'), dictionary.all_views.c.owner)).subquery('tables_and_views')
        query = select(tables.c.table_name).where(tables.c.table_name == bindparam('table_name'), tables.c.owner == bindparam('owner'))
        return query

    @reflection.cache
    def has_table(self, connection, table_name, schema=None, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link."""
        self._ensure_has_table_connection(connection)
        if not schema:
            schema = self.default_schema_name
        params = {'table_name': self.denormalize_name(table_name), 'owner': self.denormalize_schema_name(schema)}
        cursor = self._execute_reflection(connection, self._has_table_query, dblink, returns_long=False, params=params)
        return bool(cursor.scalar())

    @reflection.cache
    def has_sequence(self, connection, sequence_name, schema=None, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link."""
        if not schema:
            schema = self.default_schema_name
        query = select(dictionary.all_sequences.c.sequence_name).where(dictionary.all_sequences.c.sequence_name == self.denormalize_schema_name(sequence_name), dictionary.all_sequences.c.sequence_owner == self.denormalize_schema_name(schema))
        cursor = self._execute_reflection(connection, query, dblink, returns_long=False)
        return bool(cursor.scalar())

    def _get_default_schema_name(self, connection):
        return self.normalize_name(connection.exec_driver_sql("select sys_context( 'userenv', 'current_schema' ) from dual").scalar())

    def denormalize_schema_name(self, name):
        force = getattr(name, 'quote', None)
        if force is None and name == 'public':
            return 'PUBLIC'
        return super().denormalize_name(name)

    @reflection.flexi_cache(('schema', InternalTraversal.dp_string), ('filter_names', InternalTraversal.dp_string_list), ('dblink', InternalTraversal.dp_string))
    def _get_synonyms(self, connection, schema, filter_names, dblink, **kw):
        owner = self.denormalize_schema_name(schema or self.default_schema_name)
        has_filter_names, params = self._prepare_filter_names(filter_names)
        query = select(dictionary.all_synonyms.c.synonym_name, dictionary.all_synonyms.c.table_name, dictionary.all_synonyms.c.table_owner, dictionary.all_synonyms.c.db_link).where(dictionary.all_synonyms.c.owner == owner)
        if has_filter_names:
            query = query.where(dictionary.all_synonyms.c.synonym_name.in_(params['filter_names']))
        result = self._execute_reflection(connection, query, dblink, returns_long=False).mappings()
        return result.all()

    @lru_cache()
    def _all_objects_query(self, owner, scope, kind, has_filter_names, has_mat_views):
        query = select(dictionary.all_objects.c.object_name).select_from(dictionary.all_objects).where(dictionary.all_objects.c.owner == owner)
        if kind is ObjectKind.ANY:
            query = query.where(dictionary.all_objects.c.object_type.in_(('TABLE', 'VIEW')))
        else:
            object_type = []
            if ObjectKind.VIEW in kind:
                object_type.append('VIEW')
            if ObjectKind.MATERIALIZED_VIEW in kind and ObjectKind.TABLE not in kind:
                object_type.append('MATERIALIZED VIEW')
            if ObjectKind.TABLE in kind:
                object_type.append('TABLE')
                if has_mat_views and ObjectKind.MATERIALIZED_VIEW not in kind:
                    query = query.where(dictionary.all_objects.c.object_name.not_in(bindparam('mat_views')))
            query = query.where(dictionary.all_objects.c.object_type.in_(object_type))
        if scope is ObjectScope.DEFAULT:
            query = query.where(dictionary.all_objects.c.temporary == 'N')
        elif scope is ObjectScope.TEMPORARY:
            query = query.where(dictionary.all_objects.c.temporary == 'Y')
        if has_filter_names:
            query = query.where(dictionary.all_objects.c.object_name.in_(bindparam('filter_names')))
        return query

    @reflection.flexi_cache(('schema', InternalTraversal.dp_string), ('scope', InternalTraversal.dp_plain_obj), ('kind', InternalTraversal.dp_plain_obj), ('filter_names', InternalTraversal.dp_string_list), ('dblink', InternalTraversal.dp_string))
    def _get_all_objects(self, connection, schema, scope, kind, filter_names, dblink, **kw):
        owner = self.denormalize_schema_name(schema or self.default_schema_name)
        has_filter_names, params = self._prepare_filter_names(filter_names)
        has_mat_views = False
        if ObjectKind.TABLE in kind and ObjectKind.MATERIALIZED_VIEW not in kind:
            mat_views = self.get_materialized_view_names(connection, schema, dblink, _normalize=False, **kw)
            if mat_views:
                params['mat_views'] = mat_views
                has_mat_views = True
        query = self._all_objects_query(owner, scope, kind, has_filter_names, has_mat_views)
        result = self._execute_reflection(connection, query, dblink, returns_long=False, params=params).scalars()
        return result.all()

    def _handle_synonyms_decorator(fn):

        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            return self._handle_synonyms(fn, *args, **kwargs)
        return wrapper

    def _handle_synonyms(self, fn, connection, *args, **kwargs):
        if not kwargs.get('oracle_resolve_synonyms', False):
            return fn(self, connection, *args, **kwargs)
        original_kw = kwargs.copy()
        schema = kwargs.pop('schema', None)
        result = self._get_synonyms(connection, schema=schema, filter_names=kwargs.pop('filter_names', None), dblink=kwargs.pop('dblink', None), info_cache=kwargs.get('info_cache', None))
        dblinks_owners = defaultdict(dict)
        for row in result:
            key = (row['db_link'], row['table_owner'])
            tn = self.normalize_name(row['table_name'])
            dblinks_owners[key][tn] = row['synonym_name']
        if not dblinks_owners:
            return fn(self, connection, *args, **original_kw)
        data = {}
        for (dblink, table_owner), mapping in dblinks_owners.items():
            call_kw = {**original_kw, 'schema': table_owner, 'dblink': self.normalize_name(dblink), 'filter_names': mapping.keys()}
            call_result = fn(self, connection, *args, **call_kw)
            for (_, tn), value in call_result:
                synonym_name = self.normalize_name(mapping[tn])
                data[schema, synonym_name] = value
        return data.items()

    @reflection.cache
    def get_schema_names(self, connection, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link."""
        query = select(dictionary.all_users.c.username).order_by(dictionary.all_users.c.username)
        result = self._execute_reflection(connection, query, dblink, returns_long=False).scalars()
        return [self.normalize_name(row) for row in result]

    @reflection.cache
    def get_table_names(self, connection, schema=None, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link."""
        if schema is None:
            schema = self.default_schema_name
        den_schema = self.denormalize_schema_name(schema)
        if kw.get('oracle_resolve_synonyms', False):
            tables = select(dictionary.all_tables.c.table_name, dictionary.all_tables.c.owner, dictionary.all_tables.c.iot_name, dictionary.all_tables.c.duration, dictionary.all_tables.c.tablespace_name).union_all(select(dictionary.all_synonyms.c.synonym_name.label('table_name'), dictionary.all_synonyms.c.owner, dictionary.all_tables.c.iot_name, dictionary.all_tables.c.duration, dictionary.all_tables.c.tablespace_name).select_from(dictionary.all_tables).join(dictionary.all_synonyms, and_(dictionary.all_tables.c.table_name == dictionary.all_synonyms.c.table_name, dictionary.all_tables.c.owner == func.coalesce(dictionary.all_synonyms.c.table_owner, dictionary.all_synonyms.c.owner)))).subquery('available_tables')
        else:
            tables = dictionary.all_tables
        query = select(tables.c.table_name)
        if self.exclude_tablespaces:
            query = query.where(func.coalesce(tables.c.tablespace_name, 'no tablespace').not_in(self.exclude_tablespaces))
        query = query.where(tables.c.owner == den_schema, tables.c.iot_name.is_(null()), tables.c.duration.is_(null()))
        mat_query = select(dictionary.all_mviews.c.mview_name.label('table_name')).where(dictionary.all_mviews.c.owner == den_schema)
        query = query.except_all(mat_query) if self._supports_except_all else query.except_(mat_query)
        result = self._execute_reflection(connection, query, dblink, returns_long=False).scalars()
        return [self.normalize_name(row) for row in result]

    @reflection.cache
    def get_temp_table_names(self, connection, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link."""
        schema = self.denormalize_schema_name(self.default_schema_name)
        query = select(dictionary.all_tables.c.table_name)
        if self.exclude_tablespaces:
            query = query.where(func.coalesce(dictionary.all_tables.c.tablespace_name, 'no tablespace').not_in(self.exclude_tablespaces))
        query = query.where(dictionary.all_tables.c.owner == schema, dictionary.all_tables.c.iot_name.is_(null()), dictionary.all_tables.c.duration.is_not(null()))
        result = self._execute_reflection(connection, query, dblink, returns_long=False).scalars()
        return [self.normalize_name(row) for row in result]

    @reflection.cache
    def get_materialized_view_names(self, connection, schema=None, dblink=None, _normalize=True, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link."""
        if not schema:
            schema = self.default_schema_name
        query = select(dictionary.all_mviews.c.mview_name).where(dictionary.all_mviews.c.owner == self.denormalize_schema_name(schema))
        result = self._execute_reflection(connection, query, dblink, returns_long=False).scalars()
        if _normalize:
            return [self.normalize_name(row) for row in result]
        else:
            return result.all()

    @reflection.cache
    def get_view_names(self, connection, schema=None, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link."""
        if not schema:
            schema = self.default_schema_name
        query = select(dictionary.all_views.c.view_name).where(dictionary.all_views.c.owner == self.denormalize_schema_name(schema))
        result = self._execute_reflection(connection, query, dblink, returns_long=False).scalars()
        return [self.normalize_name(row) for row in result]

    @reflection.cache
    def get_sequence_names(self, connection, schema=None, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link."""
        if not schema:
            schema = self.default_schema_name
        query = select(dictionary.all_sequences.c.sequence_name).where(dictionary.all_sequences.c.sequence_owner == self.denormalize_schema_name(schema))
        result = self._execute_reflection(connection, query, dblink, returns_long=False).scalars()
        return [self.normalize_name(row) for row in result]

    def _value_or_raise(self, data, table, schema):
        table = self.normalize_name(str(table))
        try:
            return dict(data)[schema, table]
        except KeyError:
            raise exc.NoSuchTableError(f'{schema}.{table}' if schema else table) from None

    def _prepare_filter_names(self, filter_names):
        if filter_names:
            fn = [self.denormalize_name(name) for name in filter_names]
            return (True, {'filter_names': fn})
        else:
            return (False, {})

    @reflection.cache
    def get_table_options(self, connection, table_name, schema=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        data = self.get_multi_table_options(connection, schema=schema, filter_names=[table_name], scope=ObjectScope.ANY, kind=ObjectKind.ANY, **kw)
        return self._value_or_raise(data, table_name, schema)

    @lru_cache()
    def _table_options_query(self, owner, scope, kind, has_filter_names, has_mat_views):
        query = select(dictionary.all_tables.c.table_name, dictionary.all_tables.c.compression, dictionary.all_tables.c.compress_for).where(dictionary.all_tables.c.owner == owner)
        if has_filter_names:
            query = query.where(dictionary.all_tables.c.table_name.in_(bindparam('filter_names')))
        if scope is ObjectScope.DEFAULT:
            query = query.where(dictionary.all_tables.c.duration.is_(null()))
        elif scope is ObjectScope.TEMPORARY:
            query = query.where(dictionary.all_tables.c.duration.is_not(null()))
        if has_mat_views and ObjectKind.TABLE in kind and (ObjectKind.MATERIALIZED_VIEW not in kind):
            query = query.where(dictionary.all_tables.c.table_name.not_in(bindparam('mat_views')))
        elif ObjectKind.TABLE not in kind and ObjectKind.MATERIALIZED_VIEW in kind:
            query = query.where(dictionary.all_tables.c.table_name.in_(bindparam('mat_views')))
        return query

    @_handle_synonyms_decorator
    def get_multi_table_options(self, connection, *, schema, filter_names, scope, kind, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        owner = self.denormalize_schema_name(schema or self.default_schema_name)
        has_filter_names, params = self._prepare_filter_names(filter_names)
        has_mat_views = False
        if ObjectKind.TABLE in kind and ObjectKind.MATERIALIZED_VIEW not in kind:
            mat_views = self.get_materialized_view_names(connection, schema, dblink, _normalize=False, **kw)
            if mat_views:
                params['mat_views'] = mat_views
                has_mat_views = True
        elif ObjectKind.TABLE not in kind and ObjectKind.MATERIALIZED_VIEW in kind:
            mat_views = self.get_materialized_view_names(connection, schema, dblink, _normalize=False, **kw)
            params['mat_views'] = mat_views
        options = {}
        default = ReflectionDefaults.table_options
        if ObjectKind.TABLE in kind or ObjectKind.MATERIALIZED_VIEW in kind:
            query = self._table_options_query(owner, scope, kind, has_filter_names, has_mat_views)
            result = self._execute_reflection(connection, query, dblink, returns_long=False, params=params)
            for table, compression, compress_for in result:
                if compression == 'ENABLED':
                    data = {'oracle_compress': compress_for}
                else:
                    data = default()
                options[schema, self.normalize_name(table)] = data
        if ObjectKind.VIEW in kind and ObjectScope.DEFAULT in scope:
            for view in self.get_view_names(connection, schema, dblink, **kw):
                if not filter_names or view in filter_names:
                    options[schema, view] = default()
        return options.items()

    @reflection.cache
    def get_columns(self, connection, table_name, schema=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        data = self.get_multi_columns(connection, schema=schema, filter_names=[table_name], scope=ObjectScope.ANY, kind=ObjectKind.ANY, **kw)
        return self._value_or_raise(data, table_name, schema)

    def _run_batches(self, connection, query, dblink, returns_long, mappings, all_objects):
        each_batch = 500
        batches = list(all_objects)
        while batches:
            batch = batches[0:each_batch]
            batches[0:each_batch] = []
            result = self._execute_reflection(connection, query, dblink, returns_long=returns_long, params={'all_objects': batch})
            if mappings:
                yield from result.mappings()
            else:
                yield from result

    @lru_cache()
    def _column_query(self, owner):
        all_cols = dictionary.all_tab_cols
        all_comments = dictionary.all_col_comments
        all_ids = dictionary.all_tab_identity_cols
        if self.server_version_info >= (12,):
            add_cols = (all_cols.c.default_on_null, sql.case((all_ids.c.table_name.is_(None), sql.null()), else_=all_ids.c.generation_type + ',' + all_ids.c.identity_options).label('identity_options'))
            join_identity_cols = True
        else:
            add_cols = (sql.null().label('default_on_null'), sql.null().label('identity_options'))
            join_identity_cols = False
        query = select(all_cols.c.table_name, all_cols.c.column_name, all_cols.c.data_type, all_cols.c.char_length, all_cols.c.data_precision, all_cols.c.data_scale, all_cols.c.nullable, all_cols.c.data_default, all_comments.c.comments, all_cols.c.virtual_column, *add_cols).select_from(all_cols).outerjoin(all_comments, and_(all_cols.c.table_name == all_comments.c.table_name, all_cols.c.column_name == all_comments.c.column_name, all_cols.c.owner == all_comments.c.owner))
        if join_identity_cols:
            query = query.outerjoin(all_ids, and_(all_cols.c.table_name == all_ids.c.table_name, all_cols.c.column_name == all_ids.c.column_name, all_cols.c.owner == all_ids.c.owner))
        query = query.where(all_cols.c.table_name.in_(bindparam('all_objects')), all_cols.c.hidden_column == 'NO', all_cols.c.owner == owner).order_by(all_cols.c.table_name, all_cols.c.column_id)
        return query

    @_handle_synonyms_decorator
    def get_multi_columns(self, connection, *, schema, filter_names, scope, kind, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        owner = self.denormalize_schema_name(schema or self.default_schema_name)
        query = self._column_query(owner)
        if filter_names and kind is ObjectKind.ANY and (scope is ObjectScope.ANY):
            all_objects = [self.denormalize_name(n) for n in filter_names]
        else:
            all_objects = self._get_all_objects(connection, schema, scope, kind, filter_names, dblink, **kw)
        columns = defaultdict(list)
        result = self._run_batches(connection, query, dblink, returns_long=True, mappings=True, all_objects=all_objects)

        def maybe_int(value):
            if isinstance(value, float) and value.is_integer():
                return int(value)
            else:
                return value
        remove_size = re.compile('\\(\\d+\\)')
        for row_dict in result:
            table_name = self.normalize_name(row_dict['table_name'])
            orig_colname = row_dict['column_name']
            colname = self.normalize_name(orig_colname)
            coltype = row_dict['data_type']
            precision = maybe_int(row_dict['data_precision'])
            if coltype == 'NUMBER':
                scale = maybe_int(row_dict['data_scale'])
                if precision is None and scale == 0:
                    coltype = INTEGER()
                else:
                    coltype = NUMBER(precision, scale)
            elif coltype == 'FLOAT':
                if precision == 126:
                    coltype = DOUBLE_PRECISION()
                elif precision == 63:
                    coltype = REAL()
                else:
                    coltype = FLOAT(binary_precision=precision)
            elif coltype in ('VARCHAR2', 'NVARCHAR2', 'CHAR', 'NCHAR'):
                char_length = maybe_int(row_dict['char_length'])
                coltype = self.ischema_names.get(coltype)(char_length)
            elif 'WITH TIME ZONE' in coltype:
                coltype = TIMESTAMP(timezone=True)
            elif 'WITH LOCAL TIME ZONE' in coltype:
                coltype = TIMESTAMP(local_timezone=True)
            else:
                coltype = re.sub(remove_size, '', coltype)
                try:
                    coltype = self.ischema_names[coltype]
                except KeyError:
                    util.warn("Did not recognize type '%s' of column '%s'" % (coltype, colname))
                    coltype = sqltypes.NULLTYPE
            default = row_dict['data_default']
            if row_dict['virtual_column'] == 'YES':
                computed = dict(sqltext=default)
                default = None
            else:
                computed = None
            identity_options = row_dict['identity_options']
            if identity_options is not None:
                identity = self._parse_identity_options(identity_options, row_dict['default_on_null'])
                default = None
            else:
                identity = None
            cdict = {'name': colname, 'type': coltype, 'nullable': row_dict['nullable'] == 'Y', 'default': default, 'comment': row_dict['comments']}
            if orig_colname.lower() == orig_colname:
                cdict['quote'] = True
            if computed is not None:
                cdict['computed'] = computed
            if identity is not None:
                cdict['identity'] = identity
            columns[schema, table_name].append(cdict)
        return columns.items()

    def _parse_identity_options(self, identity_options, default_on_null):
        parts = [p.strip() for p in identity_options.split(',')]
        identity = {'always': parts[0] == 'ALWAYS', 'on_null': default_on_null == 'YES'}
        for part in parts[1:]:
            option, value = part.split(':')
            value = value.strip()
            if 'START WITH' in option:
                identity['start'] = int(value)
            elif 'INCREMENT BY' in option:
                identity['increment'] = int(value)
            elif 'MAX_VALUE' in option:
                identity['maxvalue'] = int(value)
            elif 'MIN_VALUE' in option:
                identity['minvalue'] = int(value)
            elif 'CYCLE_FLAG' in option:
                identity['cycle'] = value == 'Y'
            elif 'CACHE_SIZE' in option:
                identity['cache'] = int(value)
            elif 'ORDER_FLAG' in option:
                identity['order'] = value == 'Y'
        return identity

    @reflection.cache
    def get_table_comment(self, connection, table_name, schema=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        data = self.get_multi_table_comment(connection, schema=schema, filter_names=[table_name], scope=ObjectScope.ANY, kind=ObjectKind.ANY, **kw)
        return self._value_or_raise(data, table_name, schema)

    @lru_cache()
    def _comment_query(self, owner, scope, kind, has_filter_names):
        queries = []
        if ObjectKind.TABLE in kind or ObjectKind.VIEW in kind:
            tbl_view = select(dictionary.all_tab_comments.c.table_name, dictionary.all_tab_comments.c.comments).where(dictionary.all_tab_comments.c.owner == owner, dictionary.all_tab_comments.c.table_name.not_like('BIN$%'))
            if ObjectKind.VIEW not in kind:
                tbl_view = tbl_view.where(dictionary.all_tab_comments.c.table_type == 'TABLE')
            elif ObjectKind.TABLE not in kind:
                tbl_view = tbl_view.where(dictionary.all_tab_comments.c.table_type == 'VIEW')
            queries.append(tbl_view)
        if ObjectKind.MATERIALIZED_VIEW in kind:
            mat_view = select(dictionary.all_mview_comments.c.mview_name.label('table_name'), dictionary.all_mview_comments.c.comments).where(dictionary.all_mview_comments.c.owner == owner, dictionary.all_mview_comments.c.mview_name.not_like('BIN$%'))
            queries.append(mat_view)
        if len(queries) == 1:
            query = queries[0]
        else:
            union = sql.union_all(*queries).subquery('tables_and_views')
            query = select(union.c.table_name, union.c.comments)
        name_col = query.selected_columns.table_name
        if scope in (ObjectScope.DEFAULT, ObjectScope.TEMPORARY):
            temp = 'Y' if scope is ObjectScope.TEMPORARY else 'N'
            query = query.distinct().join(dictionary.all_objects, and_(dictionary.all_objects.c.owner == owner, dictionary.all_objects.c.object_name == name_col, dictionary.all_objects.c.temporary == temp))
        if has_filter_names:
            query = query.where(name_col.in_(bindparam('filter_names')))
        return query

    @_handle_synonyms_decorator
    def get_multi_table_comment(self, connection, *, schema, filter_names, scope, kind, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        owner = self.denormalize_schema_name(schema or self.default_schema_name)
        has_filter_names, params = self._prepare_filter_names(filter_names)
        query = self._comment_query(owner, scope, kind, has_filter_names)
        result = self._execute_reflection(connection, query, dblink, returns_long=False, params=params)
        default = ReflectionDefaults.table_comment
        ignore_mat_view = 'snapshot table for snapshot '
        return (((schema, self.normalize_name(table)), {'text': comment} if comment is not None and (not comment.startswith(ignore_mat_view)) else default()) for table, comment in result)

    @reflection.cache
    def get_indexes(self, connection, table_name, schema=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        data = self.get_multi_indexes(connection, schema=schema, filter_names=[table_name], scope=ObjectScope.ANY, kind=ObjectKind.ANY, **kw)
        return self._value_or_raise(data, table_name, schema)

    @lru_cache()
    def _index_query(self, owner):
        return select(dictionary.all_ind_columns.c.table_name, dictionary.all_ind_columns.c.index_name, dictionary.all_ind_columns.c.column_name, dictionary.all_indexes.c.index_type, dictionary.all_indexes.c.uniqueness, dictionary.all_indexes.c.compression, dictionary.all_indexes.c.prefix_length, dictionary.all_ind_columns.c.descend, dictionary.all_ind_expressions.c.column_expression).select_from(dictionary.all_ind_columns).join(dictionary.all_indexes, sql.and_(dictionary.all_ind_columns.c.index_name == dictionary.all_indexes.c.index_name, dictionary.all_ind_columns.c.index_owner == dictionary.all_indexes.c.owner)).outerjoin(dictionary.all_ind_expressions, sql.and_(dictionary.all_ind_expressions.c.index_name == dictionary.all_ind_columns.c.index_name, dictionary.all_ind_expressions.c.index_owner == dictionary.all_ind_columns.c.index_owner, dictionary.all_ind_expressions.c.column_position == dictionary.all_ind_columns.c.column_position)).where(dictionary.all_indexes.c.table_owner == owner, dictionary.all_indexes.c.table_name.in_(bindparam('all_objects'))).order_by(dictionary.all_ind_columns.c.index_name, dictionary.all_ind_columns.c.column_position)

    @reflection.flexi_cache(('schema', InternalTraversal.dp_string), ('dblink', InternalTraversal.dp_string), ('all_objects', InternalTraversal.dp_string_list))
    def _get_indexes_rows(self, connection, schema, dblink, all_objects, **kw):
        owner = self.denormalize_schema_name(schema or self.default_schema_name)
        query = self._index_query(owner)
        pks = {row_dict['constraint_name'] for row_dict in self._get_all_constraint_rows(connection, schema, dblink, all_objects, **kw) if row_dict['constraint_type'] == 'P'}
        result = self._run_batches(connection, query, dblink, returns_long=True, mappings=True, all_objects=all_objects)
        return [row_dict for row_dict in result if row_dict['index_name'] not in pks]

    @_handle_synonyms_decorator
    def get_multi_indexes(self, connection, *, schema, filter_names, scope, kind, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        all_objects = self._get_all_objects(connection, schema, scope, kind, filter_names, dblink, **kw)
        uniqueness = {'NONUNIQUE': False, 'UNIQUE': True}
        enabled = {'DISABLED': False, 'ENABLED': True}
        is_bitmap = {'BITMAP', 'FUNCTION-BASED BITMAP'}
        indexes = defaultdict(dict)
        for row_dict in self._get_indexes_rows(connection, schema, dblink, all_objects, **kw):
            index_name = self.normalize_name(row_dict['index_name'])
            table_name = self.normalize_name(row_dict['table_name'])
            table_indexes = indexes[schema, table_name]
            if index_name not in table_indexes:
                table_indexes[index_name] = index_dict = {'name': index_name, 'column_names': [], 'dialect_options': {}, 'unique': uniqueness.get(row_dict['uniqueness'], False)}
                do = index_dict['dialect_options']
                if row_dict['index_type'] in is_bitmap:
                    do['oracle_bitmap'] = True
                if enabled.get(row_dict['compression'], False):
                    do['oracle_compress'] = row_dict['prefix_length']
            else:
                index_dict = table_indexes[index_name]
            expr = row_dict['column_expression']
            if expr is not None:
                index_dict['column_names'].append(None)
                if 'expressions' in index_dict:
                    index_dict['expressions'].append(expr)
                else:
                    index_dict['expressions'] = index_dict['column_names'][:-1]
                    index_dict['expressions'].append(expr)
                if row_dict['descend'].lower() != 'asc':
                    assert row_dict['descend'].lower() == 'desc'
                    cs = index_dict.setdefault('column_sorting', {})
                    cs[expr] = ('desc',)
            else:
                assert row_dict['descend'].lower() == 'asc'
                cn = self.normalize_name(row_dict['column_name'])
                index_dict['column_names'].append(cn)
                if 'expressions' in index_dict:
                    index_dict['expressions'].append(cn)
        default = ReflectionDefaults.indexes
        return ((key, list(indexes[key].values()) if key in indexes else default()) for key in ((schema, self.normalize_name(obj_name)) for obj_name in all_objects))

    @reflection.cache
    def get_pk_constraint(self, connection, table_name, schema=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        data = self.get_multi_pk_constraint(connection, schema=schema, filter_names=[table_name], scope=ObjectScope.ANY, kind=ObjectKind.ANY, **kw)
        return self._value_or_raise(data, table_name, schema)

    @lru_cache()
    def _constraint_query(self, owner):
        local = dictionary.all_cons_columns.alias('local')
        remote = dictionary.all_cons_columns.alias('remote')
        return select(dictionary.all_constraints.c.table_name, dictionary.all_constraints.c.constraint_type, dictionary.all_constraints.c.constraint_name, local.c.column_name.label('local_column'), remote.c.table_name.label('remote_table'), remote.c.column_name.label('remote_column'), remote.c.owner.label('remote_owner'), dictionary.all_constraints.c.search_condition, dictionary.all_constraints.c.delete_rule).select_from(dictionary.all_constraints).join(local, and_(local.c.owner == dictionary.all_constraints.c.owner, dictionary.all_constraints.c.constraint_name == local.c.constraint_name)).outerjoin(remote, and_(dictionary.all_constraints.c.r_owner == remote.c.owner, dictionary.all_constraints.c.r_constraint_name == remote.c.constraint_name, or_(remote.c.position.is_(sql.null()), local.c.position == remote.c.position))).where(dictionary.all_constraints.c.owner == owner, dictionary.all_constraints.c.table_name.in_(bindparam('all_objects')), dictionary.all_constraints.c.constraint_type.in_(('R', 'P', 'U', 'C'))).order_by(dictionary.all_constraints.c.constraint_name, local.c.position)

    @reflection.flexi_cache(('schema', InternalTraversal.dp_string), ('dblink', InternalTraversal.dp_string), ('all_objects', InternalTraversal.dp_string_list))
    def _get_all_constraint_rows(self, connection, schema, dblink, all_objects, **kw):
        owner = self.denormalize_schema_name(schema or self.default_schema_name)
        query = self._constraint_query(owner)
        values = list(self._run_batches(connection, query, dblink, returns_long=False, mappings=True, all_objects=all_objects))
        return values

    @_handle_synonyms_decorator
    def get_multi_pk_constraint(self, connection, *, scope, schema, filter_names, kind, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        all_objects = self._get_all_objects(connection, schema, scope, kind, filter_names, dblink, **kw)
        primary_keys = defaultdict(dict)
        default = ReflectionDefaults.pk_constraint
        for row_dict in self._get_all_constraint_rows(connection, schema, dblink, all_objects, **kw):
            if row_dict['constraint_type'] != 'P':
                continue
            table_name = self.normalize_name(row_dict['table_name'])
            constraint_name = self.normalize_name(row_dict['constraint_name'])
            column_name = self.normalize_name(row_dict['local_column'])
            table_pk = primary_keys[schema, table_name]
            if not table_pk:
                table_pk['name'] = constraint_name
                table_pk['constrained_columns'] = [column_name]
            else:
                table_pk['constrained_columns'].append(column_name)
        return ((key, primary_keys[key] if key in primary_keys else default()) for key in ((schema, self.normalize_name(obj_name)) for obj_name in all_objects))

    @reflection.cache
    def get_foreign_keys(self, connection, table_name, schema=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        data = self.get_multi_foreign_keys(connection, schema=schema, filter_names=[table_name], scope=ObjectScope.ANY, kind=ObjectKind.ANY, **kw)
        return self._value_or_raise(data, table_name, schema)

    @_handle_synonyms_decorator
    def get_multi_foreign_keys(self, connection, *, scope, schema, filter_names, kind, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        all_objects = self._get_all_objects(connection, schema, scope, kind, filter_names, dblink, **kw)
        resolve_synonyms = kw.get('oracle_resolve_synonyms', False)
        owner = self.denormalize_schema_name(schema or self.default_schema_name)
        all_remote_owners = set()
        fkeys = defaultdict(dict)
        for row_dict in self._get_all_constraint_rows(connection, schema, dblink, all_objects, **kw):
            if row_dict['constraint_type'] != 'R':
                continue
            table_name = self.normalize_name(row_dict['table_name'])
            constraint_name = self.normalize_name(row_dict['constraint_name'])
            table_fkey = fkeys[schema, table_name]
            assert constraint_name is not None
            local_column = self.normalize_name(row_dict['local_column'])
            remote_table = self.normalize_name(row_dict['remote_table'])
            remote_column = self.normalize_name(row_dict['remote_column'])
            remote_owner_orig = row_dict['remote_owner']
            remote_owner = self.normalize_name(remote_owner_orig)
            if remote_owner_orig is not None:
                all_remote_owners.add(remote_owner_orig)
            if remote_table is None:
                if dblink and (not dblink.startswith('@')):
                    dblink = f'@{dblink}'
                util.warn(f"Got 'None' querying 'table_name' from all_cons_columns{dblink or ''} - does the user have proper rights to the table?")
                continue
            if constraint_name not in table_fkey:
                table_fkey[constraint_name] = fkey = {'name': constraint_name, 'constrained_columns': [], 'referred_schema': None, 'referred_table': remote_table, 'referred_columns': [], 'options': {}}
                if resolve_synonyms:
                    fkey['_ref_schema'] = remote_owner
                if schema is not None or remote_owner_orig != owner:
                    fkey['referred_schema'] = remote_owner
                delete_rule = row_dict['delete_rule']
                if delete_rule != 'NO ACTION':
                    fkey['options']['ondelete'] = delete_rule
            else:
                fkey = table_fkey[constraint_name]
            fkey['constrained_columns'].append(local_column)
            fkey['referred_columns'].append(remote_column)
        if resolve_synonyms and all_remote_owners:
            query = select(dictionary.all_synonyms.c.owner, dictionary.all_synonyms.c.table_name, dictionary.all_synonyms.c.table_owner, dictionary.all_synonyms.c.synonym_name).where(dictionary.all_synonyms.c.owner.in_(all_remote_owners))
            result = self._execute_reflection(connection, query, dblink, returns_long=False).mappings()
            remote_owners_lut = {}
            for row in result:
                synonym_owner = self.normalize_name(row['owner'])
                table_name = self.normalize_name(row['table_name'])
                remote_owners_lut[synonym_owner, table_name] = (row['table_owner'], row['synonym_name'])
            empty = (None, None)
            for table_fkeys in fkeys.values():
                for table_fkey in table_fkeys.values():
                    key = (table_fkey.pop('_ref_schema'), table_fkey['referred_table'])
                    remote_owner, syn_name = remote_owners_lut.get(key, empty)
                    if syn_name:
                        sn = self.normalize_name(syn_name)
                        table_fkey['referred_table'] = sn
                        if schema is not None or remote_owner != owner:
                            ro = self.normalize_name(remote_owner)
                            table_fkey['referred_schema'] = ro
                        else:
                            table_fkey['referred_schema'] = None
        default = ReflectionDefaults.foreign_keys
        return ((key, list(fkeys[key].values()) if key in fkeys else default()) for key in ((schema, self.normalize_name(obj_name)) for obj_name in all_objects))

    @reflection.cache
    def get_unique_constraints(self, connection, table_name, schema=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        data = self.get_multi_unique_constraints(connection, schema=schema, filter_names=[table_name], scope=ObjectScope.ANY, kind=ObjectKind.ANY, **kw)
        return self._value_or_raise(data, table_name, schema)

    @_handle_synonyms_decorator
    def get_multi_unique_constraints(self, connection, *, scope, schema, filter_names, kind, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        all_objects = self._get_all_objects(connection, schema, scope, kind, filter_names, dblink, **kw)
        unique_cons = defaultdict(dict)
        index_names = {row_dict['index_name'] for row_dict in self._get_indexes_rows(connection, schema, dblink, all_objects, **kw)}
        for row_dict in self._get_all_constraint_rows(connection, schema, dblink, all_objects, **kw):
            if row_dict['constraint_type'] != 'U':
                continue
            table_name = self.normalize_name(row_dict['table_name'])
            constraint_name_orig = row_dict['constraint_name']
            constraint_name = self.normalize_name(constraint_name_orig)
            column_name = self.normalize_name(row_dict['local_column'])
            table_uc = unique_cons[schema, table_name]
            assert constraint_name is not None
            if constraint_name not in table_uc:
                table_uc[constraint_name] = uc = {'name': constraint_name, 'column_names': [], 'duplicates_index': constraint_name if constraint_name_orig in index_names else None}
            else:
                uc = table_uc[constraint_name]
            uc['column_names'].append(column_name)
        default = ReflectionDefaults.unique_constraints
        return ((key, list(unique_cons[key].values()) if key in unique_cons else default()) for key in ((schema, self.normalize_name(obj_name)) for obj_name in all_objects))

    @reflection.cache
    def get_view_definition(self, connection, view_name, schema=None, dblink=None, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        if kw.get('oracle_resolve_synonyms', False):
            synonyms = self._get_synonyms(connection, schema, filter_names=[view_name], dblink=dblink)
            if synonyms:
                assert len(synonyms) == 1
                row_dict = synonyms[0]
                dblink = self.normalize_name(row_dict['db_link'])
                schema = row_dict['table_owner']
                view_name = row_dict['table_name']
        name = self.denormalize_name(view_name)
        owner = self.denormalize_schema_name(schema or self.default_schema_name)
        query = select(dictionary.all_views.c.text).where(dictionary.all_views.c.view_name == name, dictionary.all_views.c.owner == owner).union_all(select(dictionary.all_mviews.c.query).where(dictionary.all_mviews.c.mview_name == name, dictionary.all_mviews.c.owner == owner))
        rp = self._execute_reflection(connection, query, dblink, returns_long=False).scalar()
        if rp is None:
            raise exc.NoSuchTableError(f'{schema}.{view_name}' if schema else view_name)
        else:
            return rp

    @reflection.cache
    def get_check_constraints(self, connection, table_name, schema=None, include_all=False, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        data = self.get_multi_check_constraints(connection, schema=schema, filter_names=[table_name], scope=ObjectScope.ANY, include_all=include_all, kind=ObjectKind.ANY, **kw)
        return self._value_or_raise(data, table_name, schema)

    @_handle_synonyms_decorator
    def get_multi_check_constraints(self, connection, *, schema, filter_names, dblink=None, scope, kind, include_all=False, **kw):
        """Supported kw arguments are: ``dblink`` to reflect via a db link;
        ``oracle_resolve_synonyms`` to resolve names to synonyms
        """
        all_objects = self._get_all_objects(connection, schema, scope, kind, filter_names, dblink, **kw)
        not_null = re.compile('..+?. IS NOT NULL$')
        check_constraints = defaultdict(list)
        for row_dict in self._get_all_constraint_rows(connection, schema, dblink, all_objects, **kw):
            if row_dict['constraint_type'] != 'C':
                continue
            table_name = self.normalize_name(row_dict['table_name'])
            constraint_name = self.normalize_name(row_dict['constraint_name'])
            search_condition = row_dict['search_condition']
            table_checks = check_constraints[schema, table_name]
            if constraint_name is not None and (include_all or not not_null.match(search_condition)):
                table_checks.append({'name': constraint_name, 'sqltext': search_condition})
        default = ReflectionDefaults.check_constraints
        return ((key, check_constraints[key] if key in check_constraints else default()) for key in ((schema, self.normalize_name(obj_name)) for obj_name in all_objects))

    def _list_dblinks(self, connection, dblink=None):
        query = select(dictionary.all_db_links.c.db_link)
        links = self._execute_reflection(connection, query, dblink, returns_long=False).scalars()
        return [self.normalize_name(link) for link in links]