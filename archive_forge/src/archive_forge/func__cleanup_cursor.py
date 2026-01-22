from __future__ import annotations
import contextlib
import os
import weakref
from collections import defaultdict
from typing import (
import bson
from bson.codec_options import DEFAULT_CODEC_OPTIONS, TypeRegistry
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import (
from pymongo.change_stream import ChangeStream, ClusterChangeStream
from pymongo.client_options import ClientOptions
from pymongo.client_session import _EmptyServerSession
from pymongo.command_cursor import CommandCursor
from pymongo.errors import (
from pymongo.lock import _HAS_REGISTER_AT_FORK, _create_lock, _release_locks
from pymongo.pool import ConnectionClosedReason
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.server_selectors import writable_server_selector
from pymongo.server_type import SERVER_TYPE
from pymongo.settings import TopologySettings
from pymongo.topology import Topology, _ErrorContext
from pymongo.topology_description import TOPOLOGY_TYPE, TopologyDescription
from pymongo.typings import (
from pymongo.uri_parser import (
from pymongo.write_concern import DEFAULT_WRITE_CONCERN, WriteConcern
def _cleanup_cursor(self, locks_allowed: bool, cursor_id: int, address: Optional[_CursorAddress], conn_mgr: _ConnectionManager, session: Optional[ClientSession], explicit_session: bool) -> None:
    """Cleanup a cursor from cursor.close() or __del__.

        This method handles cleanup for Cursors/CommandCursors including any
        pinned connection or implicit session attached at the time the cursor
        was closed or garbage collected.

        :Parameters:
          - `locks_allowed`: True if we are allowed to acquire locks.
          - `cursor_id`: The cursor id which may be 0.
          - `address`: The _CursorAddress.
          - `conn_mgr`: The _ConnectionManager for the pinned connection or None.
          - `session`: The cursor's session.
          - `explicit_session`: True if the session was passed explicitly.
        """
    if locks_allowed:
        if cursor_id:
            if conn_mgr and conn_mgr.more_to_come:
                assert conn_mgr.conn is not None
                conn_mgr.conn.close_conn(ConnectionClosedReason.ERROR)
            else:
                self._close_cursor_now(cursor_id, address, session=session, conn_mgr=conn_mgr)
        if conn_mgr:
            conn_mgr.close()
    elif cursor_id or conn_mgr:
        self._close_cursor_soon(cursor_id, address, conn_mgr)
    if session and (not explicit_session):
        session._end_session(lock=locks_allowed)