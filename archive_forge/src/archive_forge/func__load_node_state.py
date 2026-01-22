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
def _load_node_state(self) -> bool:
    if not os.path.exists(self._node_state_path):
        return False
    try:
        with open(self._node_state_path, 'rt') as f:
            nodes = json.load(f)
    except Exception:
        return False
    if not nodes:
        return False
    self._nodes = nodes
    return True