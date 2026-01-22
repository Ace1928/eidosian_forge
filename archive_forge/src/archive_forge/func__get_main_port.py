import logging
import socket
from functools import lru_cache
from typing import Optional
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.plugins.environments.lightning import find_free_network_port
def _get_main_port(self) -> int:
    return self._comm_world.bcast(find_free_network_port(), root=0)