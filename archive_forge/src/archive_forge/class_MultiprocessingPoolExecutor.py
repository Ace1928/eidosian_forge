from __future__ import annotations
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, Future
from functools import partial
from queue import Empty, Queue
from dask import config
from dask.callbacks import local_callbacks, unpack_callbacks
from dask.core import _execute_task, flatten, get_dependencies, has_tasks, reverse_dict
from dask.order import order
from dask.typing import Key
class MultiprocessingPoolExecutor(Executor):

    def __init__(self, pool):
        self.pool = pool
        self._max_workers = len(pool._pool)

    def submit(self, fn, *args, **kwargs):
        return submit_apply_async(self.pool.apply_async, fn, *args, **kwargs)