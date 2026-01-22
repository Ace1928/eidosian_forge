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
def _update_servers(self) -> None:
    """Sync our Servers from TopologyDescription.server_descriptions.

        Hold the lock while calling this.
        """
    for address, sd in self._description.server_descriptions().items():
        if address not in self._servers:
            monitor = self._settings.monitor_class(server_description=sd, topology=self, pool=self._create_pool_for_monitor(address), topology_settings=self._settings)
            weak = None
            if self._publish_server and self._events is not None:
                weak = weakref.ref(self._events)
            server = Server(server_description=sd, pool=self._create_pool_for_server(address), monitor=monitor, topology_id=self._topology_id, listeners=self._listeners, events=weak)
            self._servers[address] = server
            server.open()
        else:
            was_writable = self._servers[address].description.is_writable
            self._servers[address].description = sd
            if was_writable != sd.is_writable:
                self._servers[address].pool.update_is_writable(sd.is_writable)
    for address, server in list(self._servers.items()):
        if not self._description.has_server(address):
            server.close()
            self._servers.pop(address)