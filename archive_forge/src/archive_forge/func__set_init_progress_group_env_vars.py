import logging
import os
import socket
from typing import Dict, List
from typing_extensions import override
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.utilities.cloud_io import get_filesystem
def _set_init_progress_group_env_vars(self) -> None:
    os.environ['MASTER_ADDR'] = str(self._main_address)
    log.debug(f'MASTER_ADDR: {os.environ['MASTER_ADDR']}')
    os.environ['MASTER_PORT'] = str(self._main_port)
    log.debug(f'MASTER_PORT: {os.environ['MASTER_PORT']}')