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
class _MongoClientErrorHandler:
    """Handle errors raised when executing an operation."""
    __slots__ = ('client', 'server_address', 'session', 'max_wire_version', 'sock_generation', 'completed_handshake', 'service_id', 'handled')

    def __init__(self, client: MongoClient, server: Server, session: Optional[ClientSession]):
        self.client = client
        self.server_address = server.description.address
        self.session = session
        self.max_wire_version = common.MIN_WIRE_VERSION
        self.sock_generation = server.pool.gen.get_overall()
        self.completed_handshake = False
        self.service_id: Optional[ObjectId] = None
        self.handled = False

    def contribute_socket(self, conn: Connection, completed_handshake: bool=True) -> None:
        """Provide socket information to the error handler."""
        self.max_wire_version = conn.max_wire_version
        self.sock_generation = conn.generation
        self.service_id = conn.service_id
        self.completed_handshake = completed_handshake

    def handle(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException]) -> None:
        if self.handled or exc_val is None:
            return
        self.handled = True
        if self.session:
            if isinstance(exc_val, ConnectionFailure):
                if self.session.in_transaction:
                    exc_val._add_error_label('TransientTransactionError')
                self.session._server_session.mark_dirty()
            if isinstance(exc_val, PyMongoError):
                if exc_val.has_error_label('TransientTransactionError') or exc_val.has_error_label('RetryableWriteError'):
                    self.session._unpin()
        err_ctx = _ErrorContext(exc_val, self.max_wire_version, self.sock_generation, self.completed_handshake, self.service_id)
        self.client._topology.handle_error(self.server_address, err_ctx)

    def __enter__(self) -> _MongoClientErrorHandler:
        return self

    def __exit__(self, exc_type: Optional[Type[Exception]], exc_val: Optional[Exception], exc_tb: Optional[TracebackType]) -> None:
        return self.handle(exc_type, exc_val)