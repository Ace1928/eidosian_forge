from __future__ import annotations
import logging
import re
from typing import cast
from typing import TYPE_CHECKING
from . import ranges
from ._psycopg_common import _PGDialect_common_psycopg
from ._psycopg_common import _PGExecutionContext_common_psycopg
from .base import INTERVAL
from .base import PGCompiler
from .base import PGIdentifierPreparer
from .base import REGCONFIG
from .json import JSON
from .json import JSONB
from .json import JSONPathType
from .types import CITEXT
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...sql import sqltypes
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class PGDialect_psycopg(_PGDialect_common_psycopg):
    driver = 'psycopg'
    supports_statement_cache = True
    supports_server_side_cursors = True
    default_paramstyle = 'pyformat'
    supports_sane_multi_rowcount = True
    execution_ctx_cls = PGExecutionContext_psycopg
    statement_compiler = PGCompiler_psycopg
    preparer = PGIdentifierPreparer_psycopg
    psycopg_version = (0, 0)
    _has_native_hstore = True
    _psycopg_adapters_map = None
    colspecs = util.update_copy(_PGDialect_common_psycopg.colspecs, {sqltypes.String: _PGString, REGCONFIG: _PGREGCONFIG, JSON: _PGJSON, CITEXT: CITEXT, sqltypes.JSON: _PGJSON, JSONB: _PGJSONB, sqltypes.JSON.JSONPathType: _PGJSONPathType, sqltypes.JSON.JSONIntIndexType: _PGJSONIntIndexType, sqltypes.JSON.JSONStrIndexType: _PGJSONStrIndexType, sqltypes.Interval: _PGInterval, INTERVAL: _PGInterval, sqltypes.Date: _PGDate, sqltypes.DateTime: _PGTimeStamp, sqltypes.Time: _PGTime, sqltypes.Integer: _PGInteger, sqltypes.SmallInteger: _PGSmallInteger, sqltypes.BigInteger: _PGBigInteger, ranges.AbstractSingleRange: _PsycopgRange, ranges.AbstractMultiRange: _PsycopgMultiRange})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.dbapi:
            m = re.match('(\\d+)\\.(\\d+)(?:\\.(\\d+))?', self.dbapi.__version__)
            if m:
                self.psycopg_version = tuple((int(x) for x in m.group(1, 2, 3) if x is not None))
            if self.psycopg_version < (3, 0, 2):
                raise ImportError('psycopg version 3.0.2 or higher is required.')
            from psycopg.adapt import AdaptersMap
            self._psycopg_adapters_map = adapters_map = AdaptersMap(self.dbapi.adapters)
            if self._native_inet_types is False:
                import psycopg.types.string
                adapters_map.register_loader('inet', psycopg.types.string.TextLoader)
                adapters_map.register_loader('cidr', psycopg.types.string.TextLoader)
            if self._json_deserializer:
                from psycopg.types.json import set_json_loads
                set_json_loads(self._json_deserializer, adapters_map)
            if self._json_serializer:
                from psycopg.types.json import set_json_dumps
                set_json_dumps(self._json_serializer, adapters_map)

    def create_connect_args(self, url):
        cargs, cparams = super().create_connect_args(url)
        if self._psycopg_adapters_map:
            cparams['context'] = self._psycopg_adapters_map
        if self.client_encoding is not None:
            cparams['client_encoding'] = self.client_encoding
        return (cargs, cparams)

    def _type_info_fetch(self, connection, name):
        from psycopg.types import TypeInfo
        return TypeInfo.fetch(connection.connection.driver_connection, name)

    def initialize(self, connection):
        super().initialize(connection)
        if not self.insert_returning:
            self.insert_executemany_returning = False
        if self.use_native_hstore:
            info = self._type_info_fetch(connection, 'hstore')
            self._has_native_hstore = info is not None
            if self._has_native_hstore:
                from psycopg.types.hstore import register_hstore
                register_hstore(info, self._psycopg_adapters_map)
                register_hstore(info, connection.connection)

    @classmethod
    def import_dbapi(cls):
        import psycopg
        return psycopg

    @classmethod
    def get_async_dialect_cls(cls, url):
        return PGDialectAsync_psycopg

    @util.memoized_property
    def _isolation_lookup(self):
        return {'READ COMMITTED': self.dbapi.IsolationLevel.READ_COMMITTED, 'READ UNCOMMITTED': self.dbapi.IsolationLevel.READ_UNCOMMITTED, 'REPEATABLE READ': self.dbapi.IsolationLevel.REPEATABLE_READ, 'SERIALIZABLE': self.dbapi.IsolationLevel.SERIALIZABLE}

    @util.memoized_property
    def _psycopg_Json(self):
        from psycopg.types import json
        return json.Json

    @util.memoized_property
    def _psycopg_Jsonb(self):
        from psycopg.types import json
        return json.Jsonb

    @util.memoized_property
    def _psycopg_TransactionStatus(self):
        from psycopg.pq import TransactionStatus
        return TransactionStatus

    @util.memoized_property
    def _psycopg_Range(self):
        from psycopg.types.range import Range
        return Range

    @util.memoized_property
    def _psycopg_Multirange(self):
        from psycopg.types.multirange import Multirange
        return Multirange

    def _do_isolation_level(self, connection, autocommit, isolation_level):
        connection.autocommit = autocommit
        connection.isolation_level = isolation_level

    def get_isolation_level(self, dbapi_connection):
        status_before = dbapi_connection.info.transaction_status
        value = super().get_isolation_level(dbapi_connection)
        if status_before == self._psycopg_TransactionStatus.IDLE:
            dbapi_connection.rollback()
        return value

    def set_isolation_level(self, dbapi_connection, level):
        if level == 'AUTOCOMMIT':
            self._do_isolation_level(dbapi_connection, autocommit=True, isolation_level=None)
        else:
            self._do_isolation_level(dbapi_connection, autocommit=False, isolation_level=self._isolation_lookup[level])

    def set_readonly(self, connection, value):
        connection.read_only = value

    def get_readonly(self, connection):
        return connection.read_only

    def on_connect(self):

        def notices(conn):
            conn.add_notice_handler(_log_notices)
        fns = [notices]
        if self.isolation_level is not None:

            def on_connect(conn):
                self.set_isolation_level(conn, self.isolation_level)
            fns.append(on_connect)

        def on_connect(conn):
            for fn in fns:
                fn(conn)
        return on_connect

    def is_disconnect(self, e, connection, cursor):
        if isinstance(e, self.dbapi.Error) and connection is not None:
            if connection.closed or connection.broken:
                return True
        return False

    def _do_prepared_twophase(self, connection, command, recover=False):
        dbapi_conn = connection.connection.dbapi_connection
        if recover or dbapi_conn.info.transaction_status != self._psycopg_TransactionStatus.IDLE:
            dbapi_conn.rollback()
        before_autocommit = dbapi_conn.autocommit
        try:
            if not before_autocommit:
                self._do_autocommit(dbapi_conn, True)
            dbapi_conn.execute(command)
        finally:
            if not before_autocommit:
                self._do_autocommit(dbapi_conn, before_autocommit)

    def do_rollback_twophase(self, connection, xid, is_prepared=True, recover=False):
        if is_prepared:
            self._do_prepared_twophase(connection, f"ROLLBACK PREPARED '{xid}'", recover=recover)
        else:
            self.do_rollback(connection.connection)

    def do_commit_twophase(self, connection, xid, is_prepared=True, recover=False):
        if is_prepared:
            self._do_prepared_twophase(connection, f"COMMIT PREPARED '{xid}'", recover=recover)
        else:
            self.do_commit(connection.connection)

    @util.memoized_property
    def _dialect_specific_select_one(self):
        return ';'