import asyncio
from datetime import datetime
import inspect
import fnmatch
import functools
import io
import json
import logging
import math
import os
import pathlib
import random
import socket
import subprocess
import sys
import tempfile
import time
import timeit
import traceback
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List, Optional
import uuid
from dataclasses import dataclass
import requests
from ray._raylet import Config
import psutil  # We must import psutil after ray because we bundle it with ray.
from ray._private import (
from ray._private.worker import RayContext
import yaml
import ray
import ray._private.gcs_utils as gcs_utils
import ray._private.memory_monitor as memory_monitor
import ray._private.services
import ray._private.utils
from ray._private.internal_api import memory_summary
from ray._private.tls_utils import generate_self_signed_tls_certs
from ray._raylet import GcsClientOptions, GlobalStateAccessor
from ray.core.generated import (
from ray.util.queue import Empty, Queue, _QueueActor
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def wait_for_num_nodes(num_nodes: int, timeout_s: int):
    curr_nodes = 0
    start = time.time()
    next_feedback = start
    max_time = start + timeout_s
    while not curr_nodes >= num_nodes:
        now = time.time()
        if now >= max_time:
            raise RuntimeError(f'Maximum wait time reached, but only {curr_nodes}/{num_nodes} nodes came up. Aborting.')
        if now >= next_feedback:
            passed = now - start
            print(f'Waiting for more nodes to come up: {curr_nodes}/{num_nodes} ({passed:.0f} seconds passed)')
            next_feedback = now + 10
        time.sleep(5)
        curr_nodes = len(ray.nodes())
    passed = time.time() - start
    print(f'Cluster is up: {curr_nodes}/{num_nodes} nodes online after {passed:.0f} seconds')