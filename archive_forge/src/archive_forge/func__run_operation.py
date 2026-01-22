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
@_csot.apply
def _run_operation(self, operation: Union[_Query, _GetMore], unpack_res: Callable, address: Optional[_Address]=None) -> Response:
    """Run a _Query/_GetMore operation and return a Response.

        :Parameters:
          - `operation`: a _Query or _GetMore object.
          - `unpack_res`: A callable that decodes the wire protocol response.
          - `address` (optional): Optional address when sending a message
            to a specific server, used for getMore.
        """
    if operation.conn_mgr:
        server = self._select_server(operation.read_preference, operation.session, address=address)
        with operation.conn_mgr.lock:
            with _MongoClientErrorHandler(self, server, operation.session) as err_handler:
                err_handler.contribute_socket(operation.conn_mgr.conn)
                return server.run_operation(operation.conn_mgr.conn, operation, operation.read_preference, self._event_listeners, unpack_res)

    def _cmd(_session: Optional[ClientSession], server: Server, conn: Connection, read_preference: _ServerMode) -> Response:
        operation.reset()
        return server.run_operation(conn, operation, read_preference, self._event_listeners, unpack_res)
    return self._retryable_read(_cmd, operation.read_preference, operation.session, address=address, retryable=isinstance(operation, message._Query))