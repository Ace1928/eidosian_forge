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
def _is_stale_error(self, address: _Address, err_ctx: _ErrorContext) -> bool:
    server = self._servers.get(address)
    if server is None:
        return True
    if server._pool.stale_generation(err_ctx.sock_generation, err_ctx.service_id):
        return True
    cur_tv = server.description.topology_version
    error = err_ctx.error
    error_tv = None
    if error and hasattr(error, 'details'):
        if isinstance(error.details, dict):
            error_tv = error.details.get('topologyVersion')
    return _is_stale_error_topology_version(cur_tv, error_tv)