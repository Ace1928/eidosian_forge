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
def _process_kill_cursors(self) -> None:
    """Process any pending kill cursors requests."""
    address_to_cursor_ids = defaultdict(list)
    pinned_cursors = []
    while True:
        try:
            address, cursor_id, conn_mgr = self.__kill_cursors_queue.pop()
        except IndexError:
            break
        if conn_mgr:
            pinned_cursors.append((address, cursor_id, conn_mgr))
        else:
            address_to_cursor_ids[address].append(cursor_id)
    for address, cursor_id, conn_mgr in pinned_cursors:
        try:
            self._cleanup_cursor(True, cursor_id, address, conn_mgr, None, False)
        except Exception as exc:
            if isinstance(exc, InvalidOperation) and self._topology._closed:
                raise
            else:
                helpers._handle_exception()
    if address_to_cursor_ids:
        topology = self._get_topology()
        for address, cursor_ids in address_to_cursor_ids.items():
            try:
                self._kill_cursors(cursor_ids, address, topology, session=None)
            except Exception as exc:
                if isinstance(exc, InvalidOperation) and self._topology._closed:
                    raise
                else:
                    helpers._handle_exception()