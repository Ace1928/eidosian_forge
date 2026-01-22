import os
from queue import Queue, Empty
from dask import config
from dask.callbacks import local_callbacks, unpack_callbacks
from dask.core import _execute_task, flatten, get_dependencies, has_tasks, reverse_dict
from dask.order import order
def apply_sync(func, args=(), kwds=None, callback=None):
    """A naive synchronous version of apply_async"""
    if kwds is None:
        kwds = {}
    res = func(*args, **kwds)
    if callback is not None:
        callback(res)