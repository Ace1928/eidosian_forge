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
class _BatchQueueActor(_QueueActor):

    async def get_batch(self, batch_size=None, total_timeout=None, first_timeout=None):
        start = timeit.default_timer()
        try:
            first = await asyncio.wait_for(self.queue.get(), first_timeout)
            batch = [first]
            if total_timeout:
                end = timeit.default_timer()
                total_timeout = max(total_timeout - (end - start), 0)
        except asyncio.TimeoutError:
            raise Empty
        if batch_size is None:
            if total_timeout is None:
                total_timeout = 0
            while True:
                try:
                    start = timeit.default_timer()
                    batch.append(await asyncio.wait_for(self.queue.get(), total_timeout))
                    if total_timeout:
                        end = timeit.default_timer()
                        total_timeout = max(total_timeout - (end - start), 0)
                except asyncio.TimeoutError:
                    break
        else:
            for _ in range(batch_size - 1):
                try:
                    start = timeit.default_timer()
                    batch.append(await asyncio.wait_for(self.queue.get(), total_timeout))
                    if total_timeout:
                        end = timeit.default_timer()
                        total_timeout = max(total_timeout - (end - start), 0)
                except asyncio.TimeoutError:
                    break
        return batch