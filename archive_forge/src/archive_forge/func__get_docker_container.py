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
def _get_docker_container(self, node_id: str) -> Optional[str]:
    self._update_status()
    node_status = self._status.get(node_id)
    if not node_status:
        return None
    return node_status['Name']