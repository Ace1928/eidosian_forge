from __future__ import annotations
import collections
import decimal
import json as _py_json
import re
import time
from . import json
from . import ranges
from .array import ARRAY as PGARRAY
from .base import _DECIMAL_TYPES
from .base import _FLOAT_TYPES
from .base import _INT_TYPES
from .base import ENUM
from .base import INTERVAL
from .base import OID
from .base import PGCompiler
from .base import PGDialect
from .base import PGExecutionContext
from .base import PGIdentifierPreparer
from .base import REGCLASS
from .base import REGCONFIG
from .types import BIT
from .types import BYTEA
from .types import CITEXT
from ... import exc
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...engine import processors
from ...sql import sqltypes
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class PGDialect_asyncpg(PGDialect):
    driver = 'asyncpg'
    supports_statement_cache = True
    supports_server_side_cursors = True
    render_bind_cast = True
    has_terminate = True
    default_paramstyle = 'numeric_dollar'
    supports_sane_multi_rowcount = False
    execution_ctx_cls = PGExecutionContext_asyncpg
    statement_compiler = PGCompiler_asyncpg
    preparer = PGIdentifierPreparer_asyncpg
    colspecs = util.update_copy(PGDialect.colspecs, {sqltypes.String: AsyncpgString, sqltypes.ARRAY: AsyncpgARRAY, BIT: AsyncpgBit, CITEXT: CITEXT, REGCONFIG: AsyncpgREGCONFIG, sqltypes.Time: AsyncpgTime, sqltypes.Date: AsyncpgDate, sqltypes.DateTime: AsyncpgDateTime, sqltypes.Interval: AsyncPgInterval, INTERVAL: AsyncPgInterval, sqltypes.Boolean: AsyncpgBoolean, sqltypes.Integer: AsyncpgInteger, sqltypes.BigInteger: AsyncpgBigInteger, sqltypes.Numeric: AsyncpgNumeric, sqltypes.Float: AsyncpgFloat, sqltypes.JSON: AsyncpgJSON, sqltypes.LargeBinary: AsyncpgByteA, json.JSONB: AsyncpgJSONB, sqltypes.JSON.JSONPathType: AsyncpgJSONPathType, sqltypes.JSON.JSONIndexType: AsyncpgJSONIndexType, sqltypes.JSON.JSONIntIndexType: AsyncpgJSONIntIndexType, sqltypes.JSON.JSONStrIndexType: AsyncpgJSONStrIndexType, sqltypes.Enum: AsyncPgEnum, OID: AsyncpgOID, REGCLASS: AsyncpgREGCLASS, sqltypes.CHAR: AsyncpgCHAR, ranges.AbstractSingleRange: _AsyncpgRange, ranges.AbstractMultiRange: _AsyncpgMultiRange})
    is_async = True
    _invalidate_schema_cache_asof = 0

    def _invalidate_schema_cache(self):
        self._invalidate_schema_cache_asof = time.time()

    @util.memoized_property
    def _dbapi_version(self):
        if self.dbapi and hasattr(self.dbapi, '__version__'):
            return tuple([int(x) for x in re.findall('(\\d+)(?:[-\\.]?|$)', self.dbapi.__version__)])
        else:
            return (99, 99, 99)

    @classmethod
    def import_dbapi(cls):
        return AsyncAdapt_asyncpg_dbapi(__import__('asyncpg'))

    @util.memoized_property
    def _isolation_lookup(self):
        return {'AUTOCOMMIT': 'autocommit', 'READ COMMITTED': 'read_committed', 'REPEATABLE READ': 'repeatable_read', 'SERIALIZABLE': 'serializable'}

    def get_isolation_level_values(self, dbapi_connection):
        return list(self._isolation_lookup)

    def set_isolation_level(self, dbapi_connection, level):
        dbapi_connection.set_isolation_level(self._isolation_lookup[level])

    def set_readonly(self, connection, value):
        connection.readonly = value

    def get_readonly(self, connection):
        return connection.readonly

    def set_deferrable(self, connection, value):
        connection.deferrable = value

    def get_deferrable(self, connection):
        return connection.deferrable

    def do_terminate(self, dbapi_connection) -> None:
        dbapi_connection.terminate()

    def create_connect_args(self, url):
        opts = url.translate_connect_args(username='user')
        multihosts, multiports = self._split_multihost_from_url(url)
        opts.update(url.query)
        if multihosts:
            assert multiports
            if len(multihosts) == 1:
                opts['host'] = multihosts[0]
                if multiports[0] is not None:
                    opts['port'] = multiports[0]
            elif not all(multihosts):
                raise exc.ArgumentError('All hosts are required to be present for asyncpg multiple host URL')
            elif not all(multiports):
                raise exc.ArgumentError('All ports are required to be present for asyncpg multiple host URL')
            else:
                opts['host'] = list(multihosts)
                opts['port'] = list(multiports)
        else:
            util.coerce_kw_type(opts, 'port', int)
        util.coerce_kw_type(opts, 'prepared_statement_cache_size', int)
        return ([], opts)

    def do_ping(self, dbapi_connection):
        dbapi_connection.ping()
        return True

    @classmethod
    def get_pool_class(cls, url):
        async_fallback = url.query.get('async_fallback', False)
        if util.asbool(async_fallback):
            return pool.FallbackAsyncAdaptedQueuePool
        else:
            return pool.AsyncAdaptedQueuePool

    def is_disconnect(self, e, connection, cursor):
        if connection:
            return connection._connection.is_closed()
        else:
            return isinstance(e, self.dbapi.InterfaceError) and 'connection is closed' in str(e)

    async def setup_asyncpg_json_codec(self, conn):
        """set up JSON codec for asyncpg.

        This occurs for all new connections and
        can be overridden by third party dialects.

        .. versionadded:: 1.4.27

        """
        asyncpg_connection = conn._connection
        deserializer = self._json_deserializer or _py_json.loads

        def _json_decoder(bin_value):
            return deserializer(bin_value.decode())
        await asyncpg_connection.set_type_codec('json', encoder=str.encode, decoder=_json_decoder, schema='pg_catalog', format='binary')

    async def setup_asyncpg_jsonb_codec(self, conn):
        """set up JSONB codec for asyncpg.

        This occurs for all new connections and
        can be overridden by third party dialects.

        .. versionadded:: 1.4.27

        """
        asyncpg_connection = conn._connection
        deserializer = self._json_deserializer or _py_json.loads

        def _jsonb_encoder(str_value):
            return b'\x01' + str_value.encode()
        deserializer = self._json_deserializer or _py_json.loads

        def _jsonb_decoder(bin_value):
            return deserializer(bin_value[1:].decode())
        await asyncpg_connection.set_type_codec('jsonb', encoder=_jsonb_encoder, decoder=_jsonb_decoder, schema='pg_catalog', format='binary')

    async def _disable_asyncpg_inet_codecs(self, conn):
        asyncpg_connection = conn._connection
        await asyncpg_connection.set_type_codec('inet', encoder=lambda s: s, decoder=lambda s: s, schema='pg_catalog', format='text')
        await asyncpg_connection.set_type_codec('cidr', encoder=lambda s: s, decoder=lambda s: s, schema='pg_catalog', format='text')

    def on_connect(self):
        """on_connect for asyncpg

        A major component of this for asyncpg is to set up type decoders at the
        asyncpg level.

        See https://github.com/MagicStack/asyncpg/issues/623 for
        notes on JSON/JSONB implementation.

        """
        super_connect = super().on_connect()

        def connect(conn):
            conn.await_(self.setup_asyncpg_json_codec(conn))
            conn.await_(self.setup_asyncpg_jsonb_codec(conn))
            if self._native_inet_types is False:
                conn.await_(self._disable_asyncpg_inet_codecs(conn))
            if super_connect is not None:
                super_connect(conn)
        return connect

    def get_driver_connection(self, connection):
        return connection._connection