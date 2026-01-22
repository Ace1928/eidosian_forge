import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple
import ray._private.ray_constants as ray_constants
from ray._private.client_mode_hook import (
from ray._private.ray_logging import setup_logger
from ray.job_config import JobConfig
from ray.util.annotations import DeveloperAPI
def _check_versions(self, conn_info: Dict[str, Any], ignore_version: bool) -> None:
    local_major_minor = f'{sys.version_info[0]}.{sys.version_info[1]}'
    if not conn_info['python_version'].startswith(local_major_minor):
        version_str = f'{local_major_minor}.{sys.version_info[2]}'
        msg = 'Python minor versions differ between client and server:' + f' client is {version_str},' + f' server is {conn_info['python_version']}'
        if ignore_version or 'RAY_IGNORE_VERSION_MISMATCH' in os.environ:
            logger.warning(msg)
        else:
            raise RuntimeError(msg)
    if CURRENT_PROTOCOL_VERSION != conn_info['protocol_version']:
        msg = 'Client Ray installation incompatible with server:' + f' client is {CURRENT_PROTOCOL_VERSION},' + f' server is {conn_info['protocol_version']}'
        if ignore_version or 'RAY_IGNORE_VERSION_MISMATCH' in os.environ:
            logger.warning(msg)
        else:
            raise RuntimeError(msg)