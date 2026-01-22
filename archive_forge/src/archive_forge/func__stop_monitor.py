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
def _stop_monitor(self):
    if self._monitor_process:
        self._monitor_process.wait(timeout=30)
        if self._monitor_process.poll() is None:
            self._monitor_process.terminate()
    self._monitor_process = None