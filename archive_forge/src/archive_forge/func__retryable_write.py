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
def _retryable_write(self, retryable: bool, func: _WriteCall[T], session: Optional[ClientSession], bulk: Optional[_Bulk]=None) -> T:
    """Execute an operation with consecutive retries if possible

        Returns func()'s return value on success. On error retries the same
        command.

        Re-raises any exception thrown by func().

        :Parameters:
          - `retryable`: if we should attempt retries (may not always be supported)
          - `func`: write call we want to execute during a session
          - `session`: Client session we will use to execute write operation
          - `bulk`: bulk abstraction to execute operations in bulk, defaults to None
        """
    with self._tmp_session(session) as s:
        return self._retry_with_session(retryable, func, s, bulk)