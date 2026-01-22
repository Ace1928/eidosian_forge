import copy
import json
import logging
import os
import subprocess
import tempfile
import time
from typing import Dict, Optional
import yaml
import ray
import ray._private.services
from ray._private import ray_constants
from ray._private.client_mode_hook import disable_client_hook
from ray._raylet import GcsClientOptions
from ray.util.annotations import DeveloperAPI
def _wait_for_node(self, node, timeout: float=30):
    """Wait until this node has appeared in the client table.

        Args:
            node (ray._private.node.Node): The node to wait for.
            timeout: The amount of time in seconds to wait before raising an
                exception.

        Raises:
            TimeoutError: An exception is raised if the timeout expires before
                the node appears in the client table.
        """
    ray._private.services.wait_for_node(node.gcs_address, node.plasma_store_socket_name, timeout)