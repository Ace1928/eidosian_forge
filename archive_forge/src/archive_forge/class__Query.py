from __future__ import annotations
import datetime
import random
import struct
from io import BytesIO as _BytesIO
from typing import (
import bson
from bson import CodecOptions, _decode_selective, _dict_to_bson, _make_c_string, encode
from bson.int64 import Int64
from bson.raw_bson import (
from bson.son import SON
from pymongo.errors import (
from pymongo.hello import HelloCompat
from pymongo.helpers import _handle_reauth
from pymongo.read_preferences import ReadPreference
from pymongo.write_concern import WriteConcern
class _Query:
    """A query operation."""
    __slots__ = ('flags', 'db', 'coll', 'ntoskip', 'spec', 'fields', 'codec_options', 'read_preference', 'limit', 'batch_size', 'name', 'read_concern', 'collation', 'session', 'client', 'allow_disk_use', '_as_command', 'exhaust')
    conn_mgr = None
    cursor_id = None

    def __init__(self, flags: int, db: str, coll: str, ntoskip: int, spec: Mapping[str, Any], fields: Optional[Mapping[str, Any]], codec_options: CodecOptions, read_preference: _ServerMode, limit: int, batch_size: int, read_concern: ReadConcern, collation: Optional[Mapping[str, Any]], session: Optional[ClientSession], client: MongoClient, allow_disk_use: Optional[bool], exhaust: bool):
        self.flags = flags
        self.db = db
        self.coll = coll
        self.ntoskip = ntoskip
        self.spec = spec
        self.fields = fields
        self.codec_options = codec_options
        self.read_preference = read_preference
        self.read_concern = read_concern
        self.limit = limit
        self.batch_size = batch_size
        self.collation = collation
        self.session = session
        self.client = client
        self.allow_disk_use = allow_disk_use
        self.name = 'find'
        self._as_command: Optional[tuple[SON[str, Any], str]] = None
        self.exhaust = exhaust

    def reset(self) -> None:
        self._as_command = None

    def namespace(self) -> str:
        return f'{self.db}.{self.coll}'

    def use_command(self, conn: Connection) -> bool:
        use_find_cmd = False
        if not self.exhaust:
            use_find_cmd = True
        elif conn.max_wire_version >= 8:
            use_find_cmd = True
        elif not self.read_concern.ok_for_legacy:
            raise ConfigurationError('read concern level of %s is not valid with a max wire version of %d.' % (self.read_concern.level, conn.max_wire_version))
        conn.validate_session(self.client, self.session)
        return use_find_cmd

    def as_command(self, conn: Connection, apply_timeout: bool=False) -> tuple[SON[str, Any], str]:
        """Return a find command document for this query."""
        if self._as_command is not None:
            return self._as_command
        explain = '$explain' in self.spec
        cmd: SON[str, Any] = _gen_find_command(self.coll, self.spec, self.fields, self.ntoskip, self.limit, self.batch_size, self.flags, self.read_concern, self.collation, self.session, self.allow_disk_use)
        if explain:
            self.name = 'explain'
            cmd = SON([('explain', cmd)])
        session = self.session
        conn.add_server_api(cmd)
        if session:
            session._apply_to(cmd, False, self.read_preference, conn)
            if not explain and (not session.in_transaction):
                session._update_read_concern(cmd, conn)
        conn.send_cluster_time(cmd, session, self.client)
        client = self.client
        if client._encrypter and (not client._encrypter._bypass_auto_encryption):
            cmd = cast(SON[str, Any], client._encrypter.encrypt(self.db, cmd, self.codec_options))
        if apply_timeout:
            conn.apply_timeout(client, cmd)
        self._as_command = (cmd, self.db)
        return self._as_command

    def get_message(self, read_preference: _ServerMode, conn: Connection, use_cmd: bool=False) -> tuple[int, bytes, int]:
        """Get a query message, possibly setting the secondaryOk bit."""
        self.read_preference = read_preference
        if read_preference.mode:
            flags = self.flags | 4
        else:
            flags = self.flags
        ns = self.namespace()
        spec = self.spec
        if use_cmd:
            spec = self.as_command(conn, apply_timeout=True)[0]
            request_id, msg, size, _ = _op_msg(0, spec, self.db, read_preference, self.codec_options, ctx=conn.compression_context)
            return (request_id, msg, size)
        ntoreturn = self.batch_size == 1 and 2 or self.batch_size
        if self.limit:
            if ntoreturn:
                ntoreturn = min(self.limit, ntoreturn)
            else:
                ntoreturn = self.limit
        if conn.is_mongos:
            assert isinstance(spec, MutableMapping)
            spec = _maybe_add_read_preference(spec, read_preference)
        return _query(flags, ns, self.ntoskip, ntoreturn, spec, None if use_cmd else self.fields, self.codec_options, ctx=conn.compression_context)