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
def _send_cluster_time(self, command: MutableMapping[str, Any], session: Optional[ClientSession]) -> None:
    topology_time = self._topology.max_cluster_time()
    session_time = session.cluster_time if session else None
    if topology_time and session_time:
        if topology_time['clusterTime'] > session_time['clusterTime']:
            cluster_time: Optional[ClusterTime] = topology_time
        else:
            cluster_time = session_time
    else:
        cluster_time = topology_time or session_time
    if cluster_time:
        command['$clusterTime'] = cluster_time