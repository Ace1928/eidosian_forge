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
@ray.remote(num_cpus=0)
class MemoryMonitorActor:

    def __init__(self, print_interval_s: float=20, record_interval_s: float=5, warning_threshold: float=0.9, n: int=10):
        """The actor that monitor the memory usage of the cluster.

            Params:
                print_interval_s: The interval where
                    memory usage is printed.
                record_interval_s: The interval where
                    memory usage is recorded.
                warning_threshold: The threshold where
                    memory warning is printed
                n: When memory usage is printed,
                    top n entries are printed.
            """
        self.print_interval_s = print_interval_s
        self.record_interval_s = record_interval_s
        self.is_running = False
        self.warning_threshold = warning_threshold
        self.monitor = memory_monitor.MemoryMonitor()
        self.n = n
        self.peak_memory_usage = 0
        self.peak_top_n_memory_usage = ''
        self._last_print_time = 0
        logging.basicConfig(level=logging.INFO)

    def ready(self):
        pass

    async def run(self):
        """Run the monitor."""
        self.is_running = True
        while self.is_running:
            now = time.time()
            used_gb, total_gb = self.monitor.get_memory_usage()
            top_n_memory_usage = memory_monitor.get_top_n_memory_usage(n=self.n)
            if used_gb > self.peak_memory_usage:
                self.peak_memory_usage = used_gb
                self.peak_top_n_memory_usage = top_n_memory_usage
            if used_gb > total_gb * self.warning_threshold:
                logging.warning(f'The memory usage is high: {used_gb / total_gb * 100}%')
            if now - self._last_print_time > self.print_interval_s:
                logging.info(f'Memory usage: {used_gb} / {total_gb}')
                logging.info(f'Top {self.n} process memory usage:')
                logging.info(top_n_memory_usage)
                self._last_print_time = now
            await asyncio.sleep(self.record_interval_s)

    async def stop_run(self):
        """Stop running the monitor.

            Returns:
                True if the monitor is stopped. False otherwise.
            """
        was_running = self.is_running
        self.is_running = False
        return was_running

    async def get_peak_memory_info(self):
        """Return the tuple of the peak memory usage and the
            top n process information during the peak memory usage.
            """
        return (self.peak_memory_usage, self.peak_top_n_memory_usage)