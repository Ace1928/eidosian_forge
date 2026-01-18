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
def wait_for_nodes(self, timeout: float=30):
    """Waits for correct number of nodes to be registered.

        This will wait until the number of live nodes in the client table
        exactly matches the number of "add_node" calls minus the number of
        "remove_node" calls that have been made on this cluster. This means
        that if a node dies without "remove_node" having been called, this will
        raise an exception.

        Args:
            timeout: The number of seconds to wait for nodes to join
                before failing.

        Raises:
            TimeoutError: An exception is raised if we time out while waiting
                for nodes to join.
        """
    start_time = time.time()
    while time.time() - start_time < timeout:
        clients = self.global_state.node_table()
        live_clients = [client for client in clients if client['Alive']]
        expected = len(self.list_all_nodes())
        if len(live_clients) == expected:
            logger.debug('All nodes registered as expected.')
            return
        else:
            logger.debug(f'{len(live_clients)} nodes are currently registered, but we are expecting {expected}')
            time.sleep(0.1)
    raise TimeoutError('Timed out while waiting for nodes to join.')