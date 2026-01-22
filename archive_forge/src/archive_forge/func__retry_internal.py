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
def _retry_internal(self, func: _WriteCall[T] | _ReadCall[T], session: Optional[ClientSession], bulk: Optional[_Bulk], is_read: bool=False, address: Optional[_Address]=None, read_pref: Optional[_ServerMode]=None, retryable: bool=False) -> T:
    """Internal retryable helper for all client transactions.

        :Parameters:
          - `func`: Callback function we want to retry
          - `session`: Client Session on which the transaction should occur
          - `bulk`: Abstraction to handle bulk write operations
          - `is_read`: If this is an exclusive read transaction, defaults to False
          - `address`: Server Address, defaults to None
          - `read_pref`: Topology of read operation, defaults to None
          - `retryable`: If the operation should be retried once, defaults to None

        :Returns:
          Output of the calling func()
        """
    return _ClientConnectionRetryable(mongo_client=self, func=func, bulk=bulk, is_read=is_read, session=session, read_pref=read_pref, address=address, retryable=retryable).run()