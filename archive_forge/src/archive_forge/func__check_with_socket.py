from __future__ import annotations
import atexit
import time
import weakref
from typing import TYPE_CHECKING, Any, Mapping, Optional, cast
from pymongo import common, periodic_executor
from pymongo._csot import MovingMinimum
from pymongo.errors import NotPrimaryError, OperationFailure, _OperationCancelled
from pymongo.hello import Hello
from pymongo.lock import _create_lock
from pymongo.periodic_executor import _shutdown_executors
from pymongo.pool import _is_faas
from pymongo.read_preferences import MovingAverage
from pymongo.server_description import ServerDescription
from pymongo.srv_resolver import _SrvResolver
def _check_with_socket(self, conn: Connection) -> tuple[Hello, float]:
    """Return (Hello, round_trip_time).

        Can raise ConnectionFailure or OperationFailure.
        """
    cluster_time = self._topology.max_cluster_time()
    start = time.monotonic()
    if conn.more_to_come:
        response = Hello(conn._next_reply(), awaitable=True)
    elif self._stream and conn.performed_handshake and self._server_description.topology_version:
        response = conn._hello(cluster_time, self._server_description.topology_version, self._settings.heartbeat_frequency)
    else:
        response = conn._hello(cluster_time, None, None)
    return (response, time.monotonic() - start)