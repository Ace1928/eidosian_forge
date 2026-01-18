import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, Optional
import yaml
import ray
from ray._private.dict import deep_update
from ray.autoscaler._private.fake_multi_node.node_provider import (
from ray.util.queue import Empty, Queue
@staticmethod
def wait_for_resources(resources: Dict[str, float], timeout: int=60):
    """Wait until Ray cluster resources are available

        Args:
            resources: Minimum resources needed before
                this function returns.
            timeout: Timeout in seconds.

        """
    timeout = time.monotonic() + timeout
    available = ray.cluster_resources()
    while any((available.get(k, 0.0) < v for k, v in resources.items())):
        if time.monotonic() > timeout:
            raise ResourcesNotReadyError(f'Timed out waiting for resources: {resources}')
        time.sleep(1)
        available = ray.cluster_resources()