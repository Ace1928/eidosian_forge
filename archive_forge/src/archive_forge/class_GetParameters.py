import os
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List, Optional, Sequence, Tuple
import yaml
import ray  # noqa: F401
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.providers import _get_node_provider
from ray.autoscaler.tags import NODE_KIND_HEAD, NODE_KIND_WORKER, TAG_RAY_NODE_KIND
import psutil
class GetParameters:

    def __init__(self, logs: bool=True, debug_state: bool=True, pip: bool=True, processes: bool=True, processes_verbose: bool=True, processes_list: Optional[List[Tuple[str, bool]]]=None):
        self.logs = logs
        self.debug_state = debug_state
        self.pip = pip
        self.processes = processes
        self.processes_verbose = processes_verbose
        self.processes_list = processes_list