import collections
import copy
import gc
import itertools
import logging
import os
import queue
import sys
import threading
import time
from multiprocessing import TimeoutError
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple
import ray
from ray._private.usage import usage_lib
from ray.util import log_once
@ray.remote(num_cpus=0)
class PoolActor:
    """Actor used to process tasks submitted to a Pool."""

    def __init__(self, initializer=None, initargs=None):
        if initializer:
            initargs = initargs or ()
            initializer(*initargs)

    def ping(self):
        pass

    def run_batch(self, func, batch):
        results = []
        for args, kwargs in batch:
            args = args or ()
            kwargs = kwargs or {}
            try:
                results.append(func(*args, **kwargs))
            except Exception as e:
                results.append(PoolTaskError(e))
        return results