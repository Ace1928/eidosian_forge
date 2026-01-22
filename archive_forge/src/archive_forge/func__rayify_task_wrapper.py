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
def _rayify_task_wrapper(key, task_info, dumps, loads, get_id, pack_exception, ray_presubmit_cbs, ray_postsubmit_cbs, ray_pretask_cbs, ray_posttask_cbs, scoped_ray_remote_args):
    """
    The core Ray-Dask task execution wrapper, to be given to the thread pool's
    `apply_async` function. Exactly the same as `execute_task`, except that it
    calls `_rayify_task` on the task instead of `_execute_task`.

    Args:
        key: The Dask graph key whose corresponding task we wish to
            execute.
        task_info: The task to execute and its dependencies.
        dumps: A result serializing function.
        loads: A task_info deserializing function.
        get_id: An ID generating function.
        pack_exception: An exception serializing function.
        ray_presubmit_cbs: Pre-task submission callbacks.
        ray_postsubmit_cbs: Post-task submission callbacks.
        ray_pretask_cbs: Pre-task execution callbacks.
        ray_posttask_cbs: Post-task execution callbacks.
        scoped_ray_remote_args: Ray task options for each key.

    Returns:
        A 3-tuple of the task's key, a literal or a Ray object reference for a
        Ray task's result, and whether the Ray task submission failed.
    """
    try:
        task, deps = loads(task_info)
        result = _rayify_task(task, key, deps, ray_presubmit_cbs, ray_postsubmit_cbs, ray_pretask_cbs, ray_posttask_cbs, scoped_ray_remote_args.get(key, {}))
        id = get_id()
        result = dumps((result, id))
        failed = False
    except BaseException as e:
        result = pack_exception(e, dumps)
        failed = True
    return (key, result, failed)