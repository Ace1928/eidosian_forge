from __future__ import annotations
import os
import queue
import random
import time
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, cast
from pymongo import _csot, common, helpers, periodic_executor
from pymongo.client_session import _ServerSession, _ServerSessionPool
from pymongo.errors import (
from pymongo.hello import Hello
from pymongo.lock import _create_lock
from pymongo.monitor import SrvMonitor
from pymongo.pool import Pool, PoolOptions
from pymongo.server import Server
from pymongo.server_description import ServerDescription
from pymongo.server_selectors import (
from pymongo.topology_description import (
def _get_replica_set_members(self, selector: Callable[[Selection], Selection]) -> set[_Address]:
    """Return set of replica set member addresses."""
    with self._lock:
        topology_type = self._description.topology_type
        if topology_type not in (TOPOLOGY_TYPE.ReplicaSetWithPrimary, TOPOLOGY_TYPE.ReplicaSetNoPrimary):
            return set()
        return {sd.address for sd in iter(selector(self._new_selection()))}