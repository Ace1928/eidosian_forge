import importlib
import inspect
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import ray.util.client_connect
from ray._private.ray_constants import (
from ray._private.utils import check_ray_client_dependencies_installed, split_address
from ray._private.worker import BaseContext
from ray._private.worker import init as ray_driver_init
from ray.job_config import JobConfig
from ray.util.annotations import Deprecated, PublicAPI
@dataclass
@PublicAPI
class ClientContext(BaseContext):
    """
    Basic context manager for a ClientBuilder connection.
    """
    dashboard_url: Optional[str]
    python_version: str
    ray_version: str
    ray_commit: str
    protocol_version: Optional[str]
    _num_clients: int
    _context_to_restore: Optional[ray.util.client.RayAPIStub]

    def __enter__(self) -> 'ClientContext':
        self._swap_context()
        return self

    def __exit__(self, *exc) -> None:
        self._disconnect_with_context(False)
        self._swap_context()

    def disconnect(self) -> None:
        self._swap_context()
        self._disconnect_with_context(True)
        self._swap_context()

    def _swap_context(self):
        if self._context_to_restore is not None:
            self._context_to_restore = ray.util.client.ray.set_context(self._context_to_restore)

    def _disconnect_with_context(self, force_disconnect: bool) -> None:
        """
        Disconnect Ray. If it's a ray client and created with `allow_multiple`,
        it will do nothing. For other cases this either disconnects from the
        remote Client Server or shuts the current driver down.
        """
        if ray.util.client.ray.is_connected():
            if ray.util.client.ray.is_default() or force_disconnect:
                ray.util.client_connect.disconnect()
        elif ray._private.worker.global_worker.node is None:
            return
        elif ray._private.worker.global_worker.node.is_head():
            logger.debug('The current Ray Cluster is scoped to this process. Disconnecting is not possible as it will shutdown the cluster.')
        else:
            ray.shutdown()