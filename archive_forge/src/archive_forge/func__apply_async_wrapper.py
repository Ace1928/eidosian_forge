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
def _apply_async_wrapper(apply_async, real_func, *extra_args, **extra_kwargs):
    """
    Wraps the given pool `apply_async` function, hotswapping `real_func` in as
    the function to be applied and adding `extra_args` and `extra_kwargs` to
    `real_func`'s call.

    Args:
        apply_async: The pool function to be wrapped.
        real_func: The real function that we wish the pool apply
            function to execute.
        *extra_args: Extra positional arguments to pass to the `real_func`.
        **extra_kwargs: Extra keyword arguments to pass to the `real_func`.

    Returns:
        A wrapper function that will ignore it's first `func` argument and
        pass `real_func` in its place. To be passed to `dask.local.get_async`.
    """

    def wrapper(func, args=(), kwds=None, callback=None):
        if not kwds:
            kwds = {}
        return apply_async(real_func, args=args + extra_args, kwds=dict(kwds, **extra_kwargs), callback=callback)
    return wrapper