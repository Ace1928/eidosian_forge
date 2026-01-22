import atexit
import collections
import datetime
import errno
import json
import logging
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from typing import Dict, Optional, Tuple, IO, AnyStr
from filelock import FileLock
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services
from ray._private import storage
from ray._raylet import GcsClient, get_session_key_from_storage
from ray._private.resource_spec import ResourceSpec
from ray._private.services import serialize_config, get_address
from ray._private.utils import open_log, try_to_create_directory, try_to_symlink
def _get_cached_port(self, port_name: str, default_port: Optional[int]=None) -> int:
    """Get a port number from a cache on this node.

        Different driver processes on a node should use the same ports for
        some purposes, e.g. exporting metrics.  This method returns a port
        number for the given port name and caches it in a file.  If the
        port isn't already cached, an unused port is generated and cached.

        Args:
            port_name: the name of the port, e.g. metrics_export_port
            default_port (Optional[int]): The port to return and cache if no
            port has already been cached for the given port_name.  If None, an
            unused port is generated and cached.
        Returns:
            port: the port number.
        """
    file_path = os.path.join(self.get_session_dir_path(), 'ports_by_node.json')
    assert port_name in ray_constants.RAY_ALLOWED_CACHED_PORTS
    ports_by_node: Dict[str, Dict[str, int]] = defaultdict(dict)
    with FileLock(file_path + '.lock'):
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump({}, f)
        with open(file_path, 'r') as f:
            ports_by_node.update(json.load(f))
        if self.unique_id in ports_by_node and port_name in ports_by_node[self.unique_id]:
            port = int(ports_by_node[self.unique_id][port_name])
        else:
            port = default_port or self._get_unused_port(set(ports_by_node[self.unique_id].values()))
            ports_by_node[self.unique_id][port_name] = port
            with open(file_path, 'w') as f:
                json.dump(ports_by_node, f)
    return port