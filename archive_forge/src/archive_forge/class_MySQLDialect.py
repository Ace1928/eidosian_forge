from the proposed insertion.   These values are normally specified using
from __future__ import annotations
from array import array as _array
from collections import defaultdict
from itertools import compress
import re
from typing import cast
from . import reflection as _reflection
from .enumerated import ENUM
from .enumerated import SET
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from .reserved_words import RESERVED_WORDS_MARIADB
from .reserved_words import RESERVED_WORDS_MYSQL
from .types import _FloatType
from .types import _IntegerType
from .types import _MatchType
from .types import _NumericType
from .types import _StringType
from .types import BIGINT
from .types import BIT
from .types import CHAR
from .types import DATETIME
from .types import DECIMAL
from .types import DOUBLE
from .types import FLOAT
from .types import INTEGER
from .types import LONGBLOB
from .types import LONGTEXT
from .types import MEDIUMBLOB
from .types import MEDIUMINT
from .types import MEDIUMTEXT
from .types import NCHAR
from .types import NUMERIC
from .types import NVARCHAR
from .types import REAL
from .types import SMALLINT
from .types import TEXT
from .types import TIME
from .types import TIMESTAMP
from .types import TINYBLOB
from .types import TINYINT
from .types import TINYTEXT
from .types import VARCHAR
from .types import YEAR
from ... import exc
from ... import literal_column
from ... import log
from ... import schema as sa_schema
from ... import sql
from ... import util
from ...engine import cursor as _cursor
from ...engine import default
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import functions
from ...sql import operators
from ...sql import roles
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql import visitors
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.compiler import SQLCompiler
from ...sql.schema import SchemaConst
from ...types import BINARY
from ...types import BLOB
from ...types import BOOLEAN
from ...types import DATE
from ...types import UUID
from ...types import VARBINARY
from ...util import topological
@log.class_logger
class MySQLDialect(default.DefaultDialect):
    """Details of the MySQL dialect.
    Not used directly in application code.
    """
    name = 'mysql'
    supports_statement_cache = True
    supports_alter = True
    supports_native_boolean = False
    max_identifier_length = 255
    max_index_name_length = 64
    max_constraint_name_length = 64
    div_is_floordiv = False
    supports_native_enum = True
    returns_native_bytes = True
    supports_sequences = False
    sequences_optional = False
    supports_for_update_of = False
    _requires_alias_for_on_duplicate_key = False
    supports_default_values = False
    supports_default_metavalue = True
    use_insertmanyvalues: bool = True
    insertmanyvalues_implicit_sentinel = InsertmanyvaluesSentinelOpts.ANY_AUTOINCREMENT
    supports_sane_rowcount = True
    supports_sane_multi_rowcount = False
    supports_multivalues_insert = True
    insert_null_pk_still_autoincrements = True
    supports_comments = True
    inline_comments = True
    default_paramstyle = 'format'
    colspecs = colspecs
    cte_follows_insert = True
    statement_compiler = MySQLCompiler
    ddl_compiler = MySQLDDLCompiler
    type_compiler_cls = MySQLTypeCompiler
    ischema_names = ischema_names
    preparer = MySQLIdentifierPreparer
    is_mariadb = False
    _mariadb_normalized_version_info = None
    _backslash_escapes = True
    _server_ansiquotes = False
    construct_arguments = [(sa_schema.Table, {'*': None}), (sql.Update, {'limit': None}), (sa_schema.PrimaryKeyConstraint, {'using': None}), (sa_schema.Index, {'using': None, 'length': None, 'prefix': None, 'with_parser': None})]

    def __init__(self, json_serializer=None, json_deserializer=None, is_mariadb=None, **kwargs):
        kwargs.pop('use_ansiquotes', None)
        default.DefaultDialect.__init__(self, **kwargs)
        self._json_serializer = json_serializer
        self._json_deserializer = json_deserializer
        self._set_mariadb(is_mariadb, None)

    def get_isolation_level_values(self, dbapi_conn):
        return ('SERIALIZABLE', 'READ UNCOMMITTED', 'READ COMMITTED', 'REPEATABLE READ')

    def set_isolation_level(self, dbapi_connection, level):
        cursor = dbapi_connection.cursor()
        cursor.execute(f'SET SESSION TRANSACTION ISOLATION LEVEL {level}')
        cursor.execute('COMMIT')
        cursor.close()

    def get_isolation_level(self, dbapi_connection):
        cursor = dbapi_connection.cursor()
        if self._is_mysql and self.server_version_info >= (5, 7, 20):
            cursor.execute('SELECT @@transaction_isolation')
        else:
            cursor.execute('SELECT @@tx_isolation')
        row = cursor.fetchone()
        if row is None:
            util.warn('Could not retrieve transaction isolation level for MySQL connection.')
            raise NotImplementedError()
        val = row[0]
        cursor.close()
        if isinstance(val, bytes):
            val = val.decode()
        return val.upper().replace('-', ' ')

    @classmethod
    def _is_mariadb_from_url(cls, url):
        dbapi = cls.import_dbapi()
        dialect = cls(dbapi=dbapi)
        cargs, cparams = dialect.create_connect_args(url)
        conn = dialect.connect(*cargs, **cparams)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT VERSION() LIKE '%MariaDB%'")
            val = cursor.fetchone()[0]
        except:
            raise
        else:
            return bool(val)
        finally:
            conn.close()

    def _get_server_version_info(self, connection):
        dbapi_con = connection.connection
        cursor = dbapi_con.cursor()
        cursor.execute('SELECT VERSION()')
        val = cursor.fetchone()[0]
        cursor.close()
        if isinstance(val, bytes):
            val = val.decode()
        return self._parse_server_version(val)

    def _parse_server_version(self, val):
        version = []
        is_mariadb = False
        r = re.compile('[.\\-+]')
        tokens = r.split(val)
        for token in tokens:
            parsed_token = re.match('^(?:(\\d+)(?:a|b|c)?|(MariaDB\\w*))$', token)
            if not parsed_token:
                continue
            elif parsed_token.group(2):
                self._mariadb_normalized_version_info = tuple(version[-3:])
                is_mariadb = True
            else:
                digit = int(parsed_token.group(1))
                version.append(digit)
        server_version_info = tuple(version)
        self._set_mariadb(server_version_info and is_mariadb, server_version_info)
        if not is_mariadb:
            self._mariadb_normalized_version_info = server_version_info
        if server_version_info < (5, 0, 2):
            raise NotImplementedError('the MySQL/MariaDB dialect supports server version info 5.0.2 and above.')
        self.server_version_info = server_version_info
        return server_version_info

    def _set_mariadb(self, is_mariadb, server_version_info):
        if is_mariadb is None:
            return
        if not is_mariadb and self.is_mariadb:
            raise exc.InvalidRequestError('MySQL version %s is not a MariaDB variant.' % ('.'.join(map(str, server_version_info)),))
        if is_mariadb:
            self.preparer = MariaDBIdentifierPreparer
            self.identifier_preparer = self.preparer(self)
            self.delete_returning = True
            self.insert_returning = True
        self.is_mariadb = is_mariadb

    def do_begin_twophase(self, connection, xid):
        connection.execute(sql.text('XA BEGIN :xid'), dict(xid=xid))

    def do_prepare_twophase(self, connection, xid):
        connection.execute(sql.text('XA END :xid'), dict(xid=xid))
        connection.execute(sql.text('XA PREPARE :xid'), dict(xid=xid))

    def do_rollback_twophase(self, connection, xid, is_prepared=True, recover=False):
        if not is_prepared:
            connection.execute(sql.text('XA END :xid'), dict(xid=xid))
        connection.execute(sql.text('XA ROLLBACK :xid'), dict(xid=xid))

    def do_commit_twophase(self, connection, xid, is_prepared=True, recover=False):
        if not is_prepared:
            self.do_prepare_twophase(connection, xid)
        connection.execute(sql.text('XA COMMIT :xid'), dict(xid=xid))

    def do_recover_twophase(self, connection):
        resultset = connection.exec_driver_sql('XA RECOVER')
        return [row['data'][0:row['gtrid_length']] for row in resultset]

    def is_disconnect(self, e, connection, cursor):
        if isinstance(e, (self.dbapi.OperationalError, self.dbapi.ProgrammingError, self.dbapi.InterfaceError)) and self._extract_error_code(e) in (1927, 2006, 2013, 2014, 2045, 2055, 4031):
            return True
        elif isinstance(e, (self.dbapi.InterfaceError, self.dbapi.InternalError)):
            return "(0, '')" in str(e)
        else:
            return False

    def _compat_fetchall(self, rp, charset=None):
        """Proxy result rows to smooth over MySQL-Python driver
        inconsistencies."""
        return [_DecodingRow(row, charset) for row in rp.fetchall()]

    def _compat_fetchone(self, rp, charset=None):
        """Proxy a result row to smooth over MySQL-Python driver
        inconsistencies."""
        row = rp.fetchone()
        if row:
            return _DecodingRow(row, charset)
        else:
            return None

    def _compat_first(self, rp, charset=None):
        """Proxy a result row to smooth over MySQL-Python driver
        inconsistencies."""
        row = rp.first()
        if row:
            return _DecodingRow(row, charset)
        else:
            return None

    def _extract_error_code(self, exception):
        raise NotImplementedError()

    def _get_default_schema_name(self, connection):
        return connection.exec_driver_sql('SELECT DATABASE()').scalar()

    @reflection.cache
    def has_table(self, connection, table_name, schema=None, **kw):
        self._ensure_has_table_connection(connection)
        if schema is None:
            schema = self.default_schema_name
        assert schema is not None
        full_name = '.'.join(self.identifier_preparer._quote_free_identifiers(schema, table_name))
        try:
            with connection.exec_driver_sql(f'DESCRIBE {full_name}', execution_options={'skip_user_error_events': True}) as rs:
                return rs.fetchone() is not None
        except exc.DBAPIError as e:
            if self._extract_error_code(e.orig) in (1146, 1049, 1051):
                return False
            raise

    @reflection.cache
    def has_sequence(self, connection, sequence_name, schema=None, **kw):
        if not self.supports_sequences:
            self._sequences_not_supported()
        if not schema:
            schema = self.default_schema_name
        cursor = connection.execute(sql.text("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='SEQUENCE' and TABLE_NAME=:name AND TABLE_SCHEMA=:schema_name"), dict(name=str(sequence_name), schema_name=str(schema)))
        return cursor.first() is not None

    def _sequences_not_supported(self):
        raise NotImplementedError('Sequences are supported only by the MariaDB series 10.3 or greater')

    @reflection.cache
    def get_sequence_names(self, connection, schema=None, **kw):
        if not self.supports_sequences:
            self._sequences_not_supported()
        if not schema:
            schema = self.default_schema_name
        cursor = connection.execute(sql.text("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='SEQUENCE' and TABLE_SCHEMA=:schema_name"), dict(schema_name=schema))
        return [row[0] for row in self._compat_fetchall(cursor, charset=self._connection_charset)]

    def initialize(self, connection):
        self._connection_charset = self._detect_charset(connection)
        default.DefaultDialect.initialize(self, connection)
        self._detect_sql_mode(connection)
        self._detect_ansiquotes(connection)
        self._detect_casing(connection)
        if self._server_ansiquotes:
            self.identifier_preparer = self.preparer(self, server_ansiquotes=self._server_ansiquotes)
        self.supports_sequences = self.is_mariadb and self.server_version_info >= (10, 3)
        self.supports_for_update_of = self._is_mysql and self.server_version_info >= (8,)
        self._needs_correct_for_88718_96365 = not self.is_mariadb and self.server_version_info >= (8,)
        self.delete_returning = self.is_mariadb and self.server_version_info >= (10, 0, 5)
        self.insert_returning = self.is_mariadb and self.server_version_info >= (10, 5)
        self._requires_alias_for_on_duplicate_key = self._is_mysql and self.server_version_info >= (8, 0, 20)
        self._warn_for_known_db_issues()

    def _warn_for_known_db_issues(self):
        if self.is_mariadb:
            mdb_version = self._mariadb_normalized_version_info
            if mdb_version > (10, 2) and mdb_version < (10, 2, 9):
                util.warn("MariaDB %r before 10.2.9 has known issues regarding CHECK constraints, which impact handling of NULL values with SQLAlchemy's boolean datatype (MDEV-13596). An additional issue prevents proper migrations of columns with CHECK constraints (MDEV-11114).  Please upgrade to MariaDB 10.2.9 or greater, or use the MariaDB 10.1 series, to avoid these issues." % (mdb_version,))

    @property
    def _support_float_cast(self):
        if not self.server_version_info:
            return False
        elif self.is_mariadb:
            return self.server_version_info >= (10, 4, 5)
        else:
            return self.server_version_info >= (8, 0, 17)

    @property
    def _is_mariadb(self):
        return self.is_mariadb

    @property
    def _is_mysql(self):
        return not self.is_mariadb

    @property
    def _is_mariadb_102(self):
        return self.is_mariadb and self._mariadb_normalized_version_info > (10, 2)

    @reflection.cache
    def get_schema_names(self, connection, **kw):
        rp = connection.exec_driver_sql('SHOW schemas')
        return [r[0] for r in rp]

    @reflection.cache
    def get_table_names(self, connection, schema=None, **kw):
        """Return a Unicode SHOW TABLES from a given schema."""
        if schema is not None:
            current_schema = schema
        else:
            current_schema = self.default_schema_name
        charset = self._connection_charset
        rp = connection.exec_driver_sql('SHOW FULL TABLES FROM %s' % self.identifier_preparer.quote_identifier(current_schema))
        return [row[0] for row in self._compat_fetchall(rp, charset=charset) if row[1] == 'BASE TABLE']

    @reflection.cache
    def get_view_names(self, connection, schema=None, **kw):
        if schema is None:
            schema = self.default_schema_name
        charset = self._connection_charset
        rp = connection.exec_driver_sql('SHOW FULL TABLES FROM %s' % self.identifier_preparer.quote_identifier(schema))
        return [row[0] for row in self._compat_fetchall(rp, charset=charset) if row[1] in ('VIEW', 'SYSTEM VIEW')]

    @reflection.cache
    def get_table_options(self, connection, table_name, schema=None, **kw):
        parsed_state = self._parsed_state_or_create(connection, table_name, schema, **kw)
        if parsed_state.table_options:
            return parsed_state.table_options
        else:
            return ReflectionDefaults.table_options()

    @reflection.cache
    def get_columns(self, connection, table_name, schema=None, **kw):
        parsed_state = self._parsed_state_or_create(connection, table_name, schema, **kw)
        if parsed_state.columns:
            return parsed_state.columns
        else:
            return ReflectionDefaults.columns()

    @reflection.cache
    def get_pk_constraint(self, connection, table_name, schema=None, **kw):
        parsed_state = self._parsed_state_or_create(connection, table_name, schema, **kw)
        for key in parsed_state.keys:
            if key['type'] == 'PRIMARY':
                cols = [s[0] for s in key['columns']]
                return {'constrained_columns': cols, 'name': None}
        return ReflectionDefaults.pk_constraint()

    @reflection.cache
    def get_foreign_keys(self, connection, table_name, schema=None, **kw):
        parsed_state = self._parsed_state_or_create(connection, table_name, schema, **kw)
        default_schema = None
        fkeys = []
        for spec in parsed_state.fk_constraints:
            ref_name = spec['table'][-1]
            ref_schema = len(spec['table']) > 1 and spec['table'][-2] or schema
            if not ref_schema:
                if default_schema is None:
                    default_schema = connection.dialect.default_schema_name
                if schema == default_schema:
                    ref_schema = schema
            loc_names = spec['local']
            ref_names = spec['foreign']
            con_kw = {}
            for opt in ('onupdate', 'ondelete'):
                if spec.get(opt, False) not in ('NO ACTION', None):
                    con_kw[opt] = spec[opt]
            fkey_d = {'name': spec['name'], 'constrained_columns': loc_names, 'referred_schema': ref_schema, 'referred_table': ref_name, 'referred_columns': ref_names, 'options': con_kw}
            fkeys.append(fkey_d)
        if self._needs_correct_for_88718_96365:
            self._correct_for_mysql_bugs_88718_96365(fkeys, connection)
        return fkeys if fkeys else ReflectionDefaults.foreign_keys()

    def _correct_for_mysql_bugs_88718_96365(self, fkeys, connection):
        if self._casing in (1, 2):

            def lower(s):
                return s.lower()
        else:

            def lower(s):
                return s
        default_schema_name = connection.dialect.default_schema_name
        col_tuples = [(lower(rec['referred_schema'] or default_schema_name), lower(rec['referred_table']), col_name) for rec in fkeys for col_name in rec['referred_columns']]
        if col_tuples:
            correct_for_wrong_fk_case = connection.execute(sql.text('\n                    select table_schema, table_name, column_name\n                    from information_schema.columns\n                    where (table_schema, table_name, lower(column_name)) in\n                    :table_data;\n                ').bindparams(sql.bindparam('table_data', expanding=True)), dict(table_data=col_tuples))
            d = defaultdict(dict)
            for schema, tname, cname in correct_for_wrong_fk_case:
                d[lower(schema), lower(tname)]['SCHEMANAME'] = schema
                d[lower(schema), lower(tname)]['TABLENAME'] = tname
                d[lower(schema), lower(tname)][cname.lower()] = cname
            for fkey in fkeys:
                rec = d[lower(fkey['referred_schema'] or default_schema_name), lower(fkey['referred_table'])]
                fkey['referred_table'] = rec['TABLENAME']
                if fkey['referred_schema'] is not None:
                    fkey['referred_schema'] = rec['SCHEMANAME']
                fkey['referred_columns'] = [rec[col.lower()] for col in fkey['referred_columns']]

    @reflection.cache
    def get_check_constraints(self, connection, table_name, schema=None, **kw):
        parsed_state = self._parsed_state_or_create(connection, table_name, schema, **kw)
        cks = [{'name': spec['name'], 'sqltext': spec['sqltext']} for spec in parsed_state.ck_constraints]
        cks.sort(key=lambda d: d['name'] or '~')
        return cks if cks else ReflectionDefaults.check_constraints()

    @reflection.cache
    def get_table_comment(self, connection, table_name, schema=None, **kw):
        parsed_state = self._parsed_state_or_create(connection, table_name, schema, **kw)
        comment = parsed_state.table_options.get(f'{self.name}_comment', None)
        if comment is not None:
            return {'text': comment}
        else:
            return ReflectionDefaults.table_comment()

    @reflection.cache
    def get_indexes(self, connection, table_name, schema=None, **kw):
        parsed_state = self._parsed_state_or_create(connection, table_name, schema, **kw)
        indexes = []
        for spec in parsed_state.keys:
            dialect_options = {}
            unique = False
            flavor = spec['type']
            if flavor == 'PRIMARY':
                continue
            if flavor == 'UNIQUE':
                unique = True
            elif flavor in ('FULLTEXT', 'SPATIAL'):
                dialect_options['%s_prefix' % self.name] = flavor
            elif flavor is None:
                pass
            else:
                self.logger.info('Converting unknown KEY type %s to a plain KEY', flavor)
                pass
            if spec['parser']:
                dialect_options['%s_with_parser' % self.name] = spec['parser']
            index_d = {}
            index_d['name'] = spec['name']
            index_d['column_names'] = [s[0] for s in spec['columns']]
            mysql_length = {s[0]: s[1] for s in spec['columns'] if s[1] is not None}
            if mysql_length:
                dialect_options['%s_length' % self.name] = mysql_length
            index_d['unique'] = unique
            if flavor:
                index_d['type'] = flavor
            if dialect_options:
                index_d['dialect_options'] = dialect_options
            indexes.append(index_d)
        indexes.sort(key=lambda d: d['name'] or '~')
        return indexes if indexes else ReflectionDefaults.indexes()

    @reflection.cache
    def get_unique_constraints(self, connection, table_name, schema=None, **kw):
        parsed_state = self._parsed_state_or_create(connection, table_name, schema, **kw)
        ucs = [{'name': key['name'], 'column_names': [col[0] for col in key['columns']], 'duplicates_index': key['name']} for key in parsed_state.keys if key['type'] == 'UNIQUE']
        ucs.sort(key=lambda d: d['name'] or '~')
        if ucs:
            return ucs
        else:
            return ReflectionDefaults.unique_constraints()

    @reflection.cache
    def get_view_definition(self, connection, view_name, schema=None, **kw):
        charset = self._connection_charset
        full_name = '.'.join(self.identifier_preparer._quote_free_identifiers(schema, view_name))
        sql = self._show_create_table(connection, None, charset, full_name=full_name)
        if sql.upper().startswith('CREATE TABLE'):
            raise exc.NoSuchTableError(full_name)
        return sql

    def _parsed_state_or_create(self, connection, table_name, schema=None, **kw):
        return self._setup_parser(connection, table_name, schema, info_cache=kw.get('info_cache', None))

    @util.memoized_property
    def _tabledef_parser(self):
        """return the MySQLTableDefinitionParser, generate if needed.

        The deferred creation ensures that the dialect has
        retrieved server version information first.

        """
        preparer = self.identifier_preparer
        return _reflection.MySQLTableDefinitionParser(self, preparer)

    @reflection.cache
    def _setup_parser(self, connection, table_name, schema=None, **kw):
        charset = self._connection_charset
        parser = self._tabledef_parser
        full_name = '.'.join(self.identifier_preparer._quote_free_identifiers(schema, table_name))
        sql = self._show_create_table(connection, None, charset, full_name=full_name)
        if parser._check_view(sql):
            columns = self._describe_table(connection, None, charset, full_name=full_name)
            sql = parser._describe_to_create(table_name, columns)
        return parser.parse(sql, charset)

    def _fetch_setting(self, connection, setting_name):
        charset = self._connection_charset
        if self.server_version_info and self.server_version_info < (5, 6):
            sql = "SHOW VARIABLES LIKE '%s'" % setting_name
            fetch_col = 1
        else:
            sql = 'SELECT @@%s' % setting_name
            fetch_col = 0
        show_var = connection.exec_driver_sql(sql)
        row = self._compat_first(show_var, charset=charset)
        if not row:
            return None
        else:
            return row[fetch_col]

    def _detect_charset(self, connection):
        raise NotImplementedError()

    def _detect_casing(self, connection):
        """Sniff out identifier case sensitivity.

        Cached per-connection. This value can not change without a server
        restart.

        """
        setting = self._fetch_setting(connection, 'lower_case_table_names')
        if setting is None:
            cs = 0
        elif setting == 'OFF':
            cs = 0
        elif setting == 'ON':
            cs = 1
        else:
            cs = int(setting)
        self._casing = cs
        return cs

    def _detect_collations(self, connection):
        """Pull the active COLLATIONS list from the server.

        Cached per-connection.
        """
        collations = {}
        charset = self._connection_charset
        rs = connection.exec_driver_sql('SHOW COLLATION')
        for row in self._compat_fetchall(rs, charset):
            collations[row[0]] = row[1]
        return collations

    def _detect_sql_mode(self, connection):
        setting = self._fetch_setting(connection, 'sql_mode')
        if setting is None:
            util.warn('Could not retrieve SQL_MODE; please ensure the MySQL user has permissions to SHOW VARIABLES')
            self._sql_mode = ''
        else:
            self._sql_mode = setting or ''

    def _detect_ansiquotes(self, connection):
        """Detect and adjust for the ANSI_QUOTES sql mode."""
        mode = self._sql_mode
        if not mode:
            mode = ''
        elif mode.isdigit():
            mode_no = int(mode)
            mode = mode_no | 4 == mode_no and 'ANSI_QUOTES' or ''
        self._server_ansiquotes = 'ANSI_QUOTES' in mode
        self._backslash_escapes = 'NO_BACKSLASH_ESCAPES' not in mode

    def _show_create_table(self, connection, table, charset=None, full_name=None):
        """Run SHOW CREATE TABLE for a ``Table``."""
        if full_name is None:
            full_name = self.identifier_preparer.format_table(table)
        st = 'SHOW CREATE TABLE %s' % full_name
        rp = None
        try:
            rp = connection.execution_options(skip_user_error_events=True).exec_driver_sql(st)
        except exc.DBAPIError as e:
            if self._extract_error_code(e.orig) == 1146:
                raise exc.NoSuchTableError(full_name) from e
            else:
                raise
        row = self._compat_first(rp, charset=charset)
        if not row:
            raise exc.NoSuchTableError(full_name)
        return row[1].strip()

    def _describe_table(self, connection, table, charset=None, full_name=None):
        """Run DESCRIBE for a ``Table`` and return processed rows."""
        if full_name is None:
            full_name = self.identifier_preparer.format_table(table)
        st = 'DESCRIBE %s' % full_name
        rp, rows = (None, None)
        try:
            try:
                rp = connection.execution_options(skip_user_error_events=True).exec_driver_sql(st)
            except exc.DBAPIError as e:
                code = self._extract_error_code(e.orig)
                if code == 1146:
                    raise exc.NoSuchTableError(full_name) from e
                elif code == 1356:
                    raise exc.UnreflectableTableError('Table or view named %s could not be reflected: %s' % (full_name, e)) from e
                else:
                    raise
            rows = self._compat_fetchall(rp, charset=charset)
        finally:
            if rp:
                rp.close()
        return rows