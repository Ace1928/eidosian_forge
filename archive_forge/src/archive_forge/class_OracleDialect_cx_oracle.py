from __future__ import annotations
import decimal
import random
import re
from . import base as oracle
from .base import OracleCompiler
from .base import OracleDialect
from .base import OracleExecutionContext
from .types import _OracleDateLiteralRender
from ... import exc
from ... import util
from ...engine import cursor as _cursor
from ...engine import interfaces
from ...engine import processors
from ...sql import sqltypes
from ...sql._typing import is_sql_compiler
class OracleDialect_cx_oracle(OracleDialect):
    supports_statement_cache = True
    execution_ctx_cls = OracleExecutionContext_cx_oracle
    statement_compiler = OracleCompiler_cx_oracle
    supports_sane_rowcount = True
    supports_sane_multi_rowcount = True
    insert_executemany_returning = True
    insert_executemany_returning_sort_by_parameter_order = True
    update_executemany_returning = True
    delete_executemany_returning = True
    bind_typing = interfaces.BindTyping.SETINPUTSIZES
    driver = 'cx_oracle'
    colspecs = util.update_copy(OracleDialect.colspecs, {sqltypes.TIMESTAMP: _CXOracleTIMESTAMP, sqltypes.Numeric: _OracleNumeric, sqltypes.Float: _OracleNumeric, oracle.BINARY_FLOAT: _OracleBINARY_FLOAT, oracle.BINARY_DOUBLE: _OracleBINARY_DOUBLE, sqltypes.Integer: _OracleInteger, oracle.NUMBER: _OracleNUMBER, sqltypes.Date: _CXOracleDate, sqltypes.LargeBinary: _OracleBinary, sqltypes.Boolean: oracle._OracleBoolean, sqltypes.Interval: _OracleInterval, oracle.INTERVAL: _OracleInterval, sqltypes.Text: _OracleText, sqltypes.String: _OracleString, sqltypes.UnicodeText: _OracleUnicodeTextCLOB, sqltypes.CHAR: _OracleChar, sqltypes.NCHAR: _OracleNChar, sqltypes.Enum: _OracleEnum, oracle.LONG: _OracleLong, oracle.RAW: _OracleRaw, sqltypes.Unicode: _OracleUnicodeStringCHAR, sqltypes.NVARCHAR: _OracleUnicodeStringNCHAR, sqltypes.Uuid: _OracleUUID, oracle.NCLOB: _OracleUnicodeTextNCLOB, oracle.ROWID: _OracleRowid})
    execute_sequence_format = list
    _cx_oracle_threaded = None
    _cursor_var_unicode_kwargs = util.immutabledict()

    @util.deprecated_params(threaded=('1.3', "The 'threaded' parameter to the cx_oracle/oracledb dialect is deprecated as a dialect-level argument, and will be removed in a future release.  As of version 1.3, it defaults to False rather than True.  The 'threaded' option can be passed to cx_Oracle directly in the URL query string passed to :func:`_sa.create_engine`."))
    def __init__(self, auto_convert_lobs=True, coerce_to_decimal=True, arraysize=None, encoding_errors=None, threaded=None, **kwargs):
        OracleDialect.__init__(self, **kwargs)
        self.arraysize = arraysize
        self.encoding_errors = encoding_errors
        if encoding_errors:
            self._cursor_var_unicode_kwargs = {'encodingErrors': encoding_errors}
        if threaded is not None:
            self._cx_oracle_threaded = threaded
        self.auto_convert_lobs = auto_convert_lobs
        self.coerce_to_decimal = coerce_to_decimal
        if self._use_nchar_for_unicode:
            self.colspecs = self.colspecs.copy()
            self.colspecs[sqltypes.Unicode] = _OracleUnicodeStringNCHAR
            self.colspecs[sqltypes.UnicodeText] = _OracleUnicodeTextNCLOB
        dbapi_module = self.dbapi
        self._load_version(dbapi_module)
        if dbapi_module is not None:
            self.include_set_input_sizes = {dbapi_module.DATETIME, dbapi_module.DB_TYPE_NVARCHAR, dbapi_module.DB_TYPE_RAW, dbapi_module.NCLOB, dbapi_module.CLOB, dbapi_module.LOB, dbapi_module.BLOB, dbapi_module.NCHAR, dbapi_module.FIXED_NCHAR, dbapi_module.FIXED_CHAR, dbapi_module.TIMESTAMP, int, dbapi_module.NATIVE_FLOAT}
            self._paramval = lambda value: value.getvalue()

    def _load_version(self, dbapi_module):
        version = (0, 0, 0)
        if dbapi_module is not None:
            m = re.match('(\\d+)\\.(\\d+)(?:\\.(\\d+))?', dbapi_module.version)
            if m:
                version = tuple((int(x) for x in m.group(1, 2, 3) if x is not None))
        self.cx_oracle_ver = version
        if self.cx_oracle_ver < (8,) and self.cx_oracle_ver > (0, 0, 0):
            raise exc.InvalidRequestError('cx_Oracle version 8 and above are supported')

    @classmethod
    def import_dbapi(cls):
        import cx_Oracle
        return cx_Oracle

    def initialize(self, connection):
        super().initialize(connection)
        self._detect_decimal_char(connection)

    def get_isolation_level(self, dbapi_connection):
        with dbapi_connection.cursor() as cursor:
            outval = cursor.var(str)
            cursor.execute('\n                begin\n                   :trans_id := dbms_transaction.local_transaction_id( TRUE );\n                end;\n                ', {'trans_id': outval})
            trans_id = outval.getvalue()
            xidusn, xidslot, xidsqn = trans_id.split('.', 2)
            cursor.execute("SELECT CASE BITAND(t.flag, POWER(2, 28)) WHEN 0 THEN 'READ COMMITTED' ELSE 'SERIALIZABLE' END AS isolation_level FROM v$transaction t WHERE (t.xidusn, t.xidslot, t.xidsqn) = ((:xidusn, :xidslot, :xidsqn))", {'xidusn': xidusn, 'xidslot': xidslot, 'xidsqn': xidsqn})
            row = cursor.fetchone()
            if row is None:
                raise exc.InvalidRequestError('could not retrieve isolation level')
            result = row[0]
        return result

    def get_isolation_level_values(self, dbapi_connection):
        return super().get_isolation_level_values(dbapi_connection) + ['AUTOCOMMIT']

    def set_isolation_level(self, dbapi_connection, level):
        if level == 'AUTOCOMMIT':
            dbapi_connection.autocommit = True
        else:
            dbapi_connection.autocommit = False
            dbapi_connection.rollback()
            with dbapi_connection.cursor() as cursor:
                cursor.execute(f'ALTER SESSION SET ISOLATION_LEVEL={level}')

    def _detect_decimal_char(self, connection):
        dbapi_connection = connection.connection
        with dbapi_connection.cursor() as cursor:

            def output_type_handler(cursor, name, defaultType, size, precision, scale):
                return cursor.var(self.dbapi.STRING, 255, arraysize=cursor.arraysize)
            cursor.outputtypehandler = output_type_handler
            cursor.execute('SELECT 1.1 FROM DUAL')
            value = cursor.fetchone()[0]
            decimal_char = value.lstrip('0')[1]
            assert not decimal_char[0].isdigit()
        self._decimal_char = decimal_char
        if self._decimal_char != '.':
            _detect_decimal = self._detect_decimal
            _to_decimal = self._to_decimal
            self._detect_decimal = lambda value: _detect_decimal(value.replace(self._decimal_char, '.'))
            self._to_decimal = lambda value: _to_decimal(value.replace(self._decimal_char, '.'))

    def _detect_decimal(self, value):
        if '.' in value:
            return self._to_decimal(value)
        else:
            return int(value)
    _to_decimal = decimal.Decimal

    def _generate_connection_outputtype_handler(self):
        """establish the default outputtypehandler established at the
        connection level.

        """
        dialect = self
        cx_Oracle = dialect.dbapi
        number_handler = _OracleNUMBER(asdecimal=True)._cx_oracle_outputtypehandler(dialect)
        float_handler = _OracleNUMBER(asdecimal=False)._cx_oracle_outputtypehandler(dialect)

        def output_type_handler(cursor, name, default_type, size, precision, scale):
            if default_type == cx_Oracle.NUMBER and default_type is not cx_Oracle.NATIVE_FLOAT:
                if not dialect.coerce_to_decimal:
                    return None
                elif precision == 0 and scale in (0, -127):
                    return cursor.var(cx_Oracle.STRING, 255, outconverter=dialect._detect_decimal, arraysize=cursor.arraysize)
                elif precision and scale > 0:
                    return number_handler(cursor, name, default_type, size, precision, scale)
                else:
                    return float_handler(cursor, name, default_type, size, precision, scale)
            elif dialect._cursor_var_unicode_kwargs and default_type in (cx_Oracle.STRING, cx_Oracle.FIXED_CHAR) and (default_type is not cx_Oracle.CLOB) and (default_type is not cx_Oracle.NCLOB):
                return cursor.var(str, size, cursor.arraysize, **dialect._cursor_var_unicode_kwargs)
            elif dialect.auto_convert_lobs and default_type in (cx_Oracle.CLOB, cx_Oracle.NCLOB):
                return cursor.var(cx_Oracle.DB_TYPE_NVARCHAR, _CX_ORACLE_MAGIC_LOB_SIZE, cursor.arraysize, **dialect._cursor_var_unicode_kwargs)
            elif dialect.auto_convert_lobs and default_type in (cx_Oracle.BLOB,):
                return cursor.var(cx_Oracle.DB_TYPE_RAW, _CX_ORACLE_MAGIC_LOB_SIZE, cursor.arraysize)
        return output_type_handler

    def on_connect(self):
        output_type_handler = self._generate_connection_outputtype_handler()

        def on_connect(conn):
            conn.outputtypehandler = output_type_handler
        return on_connect

    def create_connect_args(self, url):
        opts = dict(url.query)
        for opt in ('use_ansi', 'auto_convert_lobs'):
            if opt in opts:
                util.warn_deprecated(f'{self.driver} dialect option {opt!r} should only be passed to create_engine directly, not within the URL string', version='1.3')
                util.coerce_kw_type(opts, opt, bool)
                setattr(self, opt, opts.pop(opt))
        database = url.database
        service_name = opts.pop('service_name', None)
        if database or service_name:
            port = url.port
            if port:
                port = int(port)
            else:
                port = 1521
            if database and service_name:
                raise exc.InvalidRequestError('"service_name" option shouldn\'t be used with a "database" part of the url')
            if database:
                makedsn_kwargs = {'sid': database}
            if service_name:
                makedsn_kwargs = {'service_name': service_name}
            dsn = self.dbapi.makedsn(url.host, port, **makedsn_kwargs)
        else:
            dsn = url.host
        if dsn is not None:
            opts['dsn'] = dsn
        if url.password is not None:
            opts['password'] = url.password
        if url.username is not None:
            opts['user'] = url.username
        if self._cx_oracle_threaded is not None:
            opts.setdefault('threaded', self._cx_oracle_threaded)

        def convert_cx_oracle_constant(value):
            if isinstance(value, str):
                try:
                    int_val = int(value)
                except ValueError:
                    value = value.upper()
                    return getattr(self.dbapi, value)
                else:
                    return int_val
            else:
                return value
        util.coerce_kw_type(opts, 'mode', convert_cx_oracle_constant)
        util.coerce_kw_type(opts, 'threaded', bool)
        util.coerce_kw_type(opts, 'events', bool)
        util.coerce_kw_type(opts, 'purity', convert_cx_oracle_constant)
        return ([], opts)

    def _get_server_version_info(self, connection):
        return tuple((int(x) for x in connection.connection.version.split('.')))

    def is_disconnect(self, e, connection, cursor):
        error, = e.args
        if isinstance(e, (self.dbapi.InterfaceError, self.dbapi.DatabaseError)) and 'not connected' in str(e):
            return True
        if hasattr(error, 'code') and error.code in {28, 3114, 3113, 3135, 1033, 2396}:
            return True
        if re.match('^(?:DPI-1010|DPI-1080|DPY-1001|DPY-4011)', str(e)):
            return True
        return False

    def create_xid(self):
        """create a two-phase transaction ID.

        this id will be passed to do_begin_twophase(), do_rollback_twophase(),
        do_commit_twophase().  its format is unspecified.

        """
        id_ = random.randint(0, 2 ** 128)
        return (4660, '%032x' % id_, '%032x' % 9)

    def do_executemany(self, cursor, statement, parameters, context=None):
        if isinstance(parameters, tuple):
            parameters = list(parameters)
        cursor.executemany(statement, parameters)

    def do_begin_twophase(self, connection, xid):
        connection.connection.begin(*xid)
        connection.connection.info['cx_oracle_xid'] = xid

    def do_prepare_twophase(self, connection, xid):
        result = connection.connection.prepare()
        connection.info['cx_oracle_prepared'] = result

    def do_rollback_twophase(self, connection, xid, is_prepared=True, recover=False):
        self.do_rollback(connection.connection)

    def do_commit_twophase(self, connection, xid, is_prepared=True, recover=False):
        if not is_prepared:
            self.do_commit(connection.connection)
        else:
            if recover:
                raise NotImplementedError('2pc recovery not implemented for cx_Oracle')
            oci_prepared = connection.info['cx_oracle_prepared']
            if oci_prepared:
                self.do_commit(connection.connection)

    def do_set_input_sizes(self, cursor, list_of_tuples, context):
        if self.positional:
            cursor.setinputsizes(*[dbtype for key, dbtype, sqltype in list_of_tuples])
        else:
            collection = ((key, dbtype) for key, dbtype, sqltype in list_of_tuples if dbtype)
            cursor.setinputsizes(**{key: dbtype for key, dbtype in collection})

    def do_recover_twophase(self, connection):
        raise NotImplementedError('recover two phase query for cx_Oracle not implemented')