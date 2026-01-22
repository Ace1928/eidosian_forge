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
class _GetMore:
    """A getmore operation."""
    __slots__ = ('db', 'coll', 'ntoreturn', 'cursor_id', 'max_await_time_ms', 'codec_options', 'read_preference', 'session', 'client', 'conn_mgr', '_as_command', 'exhaust', 'comment')
    name = 'getMore'

    def __init__(self, db: str, coll: str, ntoreturn: int, cursor_id: int, codec_options: CodecOptions, read_preference: _ServerMode, session: Optional[ClientSession], client: MongoClient, max_await_time_ms: Optional[int], conn_mgr: Any, exhaust: bool, comment: Any):
        self.db = db
        self.coll = coll
        self.ntoreturn = ntoreturn
        self.cursor_id = cursor_id
        self.codec_options = codec_options
        self.read_preference = read_preference
        self.session = session
        self.client = client
        self.max_await_time_ms = max_await_time_ms
        self.conn_mgr = conn_mgr
        self._as_command: Optional[tuple[SON[str, Any], str]] = None
        self.exhaust = exhaust
        self.comment = comment

    def reset(self) -> None:
        self._as_command = None

    def namespace(self) -> str:
        return f'{self.db}.{self.coll}'

    def use_command(self, conn: Connection) -> bool:
        use_cmd = False
        if not self.exhaust:
            use_cmd = True
        elif conn.max_wire_version >= 8:
            use_cmd = True
        conn.validate_session(self.client, self.session)
        return use_cmd

    def as_command(self, conn: Connection, apply_timeout: bool=False) -> tuple[SON[str, Any], str]:
        """Return a getMore command document for this query."""
        if self._as_command is not None:
            return self._as_command
        cmd: SON[str, Any] = _gen_get_more_command(self.cursor_id, self.coll, self.ntoreturn, self.max_await_time_ms, self.comment, conn)
        if self.session:
            self.session._apply_to(cmd, False, self.read_preference, conn)
        conn.add_server_api(cmd)
        conn.send_cluster_time(cmd, self.session, self.client)
        client = self.client
        if client._encrypter and (not client._encrypter._bypass_auto_encryption):
            cmd = cast(SON[str, Any], client._encrypter.encrypt(self.db, cmd, self.codec_options))
        if apply_timeout:
            conn.apply_timeout(client, cmd=None)
        self._as_command = (cmd, self.db)
        return self._as_command

    def get_message(self, dummy0: Any, conn: Connection, use_cmd: bool=False) -> Union[tuple[int, bytes, int], tuple[int, bytes]]:
        """Get a getmore message."""
        ns = self.namespace()
        ctx = conn.compression_context
        if use_cmd:
            spec = self.as_command(conn, apply_timeout=True)[0]
            if self.conn_mgr and self.exhaust:
                flags = _OpMsg.EXHAUST_ALLOWED
            else:
                flags = 0
            request_id, msg, size, _ = _op_msg(flags, spec, self.db, None, self.codec_options, ctx=conn.compression_context)
            return (request_id, msg, size)
        return _get_more(ns, self.ntoreturn, self.cursor_id, ctx)