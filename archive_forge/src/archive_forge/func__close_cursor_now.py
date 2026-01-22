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
def _close_cursor_now(self, cursor_id: int, address: Optional[_CursorAddress], session: Optional[ClientSession]=None, conn_mgr: Optional[_ConnectionManager]=None) -> None:
    """Send a kill cursors message with the given id.

        The cursor is closed synchronously on the current thread.
        """
    if not isinstance(cursor_id, int):
        raise TypeError('cursor_id must be an instance of int')
    try:
        if conn_mgr:
            with conn_mgr.lock:
                assert address is not None
                assert conn_mgr.conn is not None
                self._kill_cursor_impl([cursor_id], address, session, conn_mgr.conn)
        else:
            self._kill_cursors([cursor_id], address, self._get_topology(), session)
    except PyMongoError:
        self._close_cursor_soon(cursor_id, address)