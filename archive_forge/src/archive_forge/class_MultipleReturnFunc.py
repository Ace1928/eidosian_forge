import atexit
import threading
from collections import defaultdict
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import Optional
import ray
import dask
from dask.core import istask, ishashable, _execute_task
from dask.system import CPU_COUNT
from dask.threaded import pack_exception, _thread_get_id
from ray.util.dask.callbacks import local_ray_callbacks, unpack_ray_callbacks
from ray.util.dask.common import unpack_object_refs
from ray.util.dask.scheduler_utils import get_async, apply_sync
@dataclass
class MultipleReturnFunc:
    func: callable
    num_returns: int

    def __call__(self, *args, **kwargs):
        returns = self.func(*args, **kwargs)
        if isinstance(returns, dict) or isinstance(returns, OrderedDict):
            returns = [returns[k] for k in range(len(returns))]
        return returns