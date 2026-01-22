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
def _update_docker_compose_config(self):
    config = copy.deepcopy(DOCKER_COMPOSE_SKELETON)
    config['services'] = {}
    for node_id, node in self._nodes.items():
        config['services'][node_id] = node['node_spec']
    with open(self._docker_compose_config_path, 'wt') as f:
        yaml.safe_dump(config, f)