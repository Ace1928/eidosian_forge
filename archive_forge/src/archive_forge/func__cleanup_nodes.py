import copy
import json
import logging
import os
import subprocess
import sys
import time
from threading import RLock
from types import ModuleType
from typing import Any, Dict, Optional
import yaml
import ray
import ray._private.ray_constants as ray_constants
from ray.autoscaler._private.fake_multi_node.command_runner import (
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def _cleanup_nodes(self):
    for node_id, timestamp in list(self._possibly_terminated_nodes.items()):
        if time.monotonic() > timestamp + self._cleanup_interval:
            if not self._is_docker_running(node_id):
                self._nodes.pop(node_id, None)
            self._possibly_terminated_nodes.pop(node_id, None)
    self._save_node_state()