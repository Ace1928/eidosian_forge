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
def wait_until_succeeded_without_exception(func, exceptions, *args, timeout_ms=1000, retry_interval_ms=100, raise_last_ex=False):
    """A helper function that waits until a given function
        completes without exceptions.

    Args:
        func: A function to run.
        exceptions: Exceptions that are supposed to occur.
        args: arguments to pass for a given func
        timeout_ms: Maximum timeout in milliseconds.
        retry_interval_ms: Retry interval in milliseconds.
        raise_last_ex: Raise the last exception when timeout.

    Return:
        Whether exception occurs within a timeout.
    """
    if type(exceptions) != tuple:
        raise Exception('exceptions arguments should be given as a tuple')
    time_elapsed = 0
    start = time.time()
    last_ex = None
    while time_elapsed <= timeout_ms:
        try:
            func(*args)
            return True
        except exceptions as ex:
            last_ex = ex
            time_elapsed = (time.time() - start) * 1000
            time.sleep(retry_interval_ms / 1000.0)
    if raise_last_ex:
        ex_stack = traceback.format_exception(type(last_ex), last_ex, last_ex.__traceback__) if last_ex else []
        ex_stack = ''.join(ex_stack)
        raise Exception(f'Timed out while testing, {ex_stack}')
    return False