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
def _convert_to_ray_batched_calls_if_needed(self, func: Callable) -> Callable:
    """Convert joblib's BatchedCalls to RayBatchedCalls for ObjectRef caching.

        This converts joblib's BatchedCalls callable, which is a collection of
        functions with their args and kwargs to be ran sequentially in an
        Actor, to a RayBatchedCalls callable, which provides identical
        functionality in addition to a method which ensures that common
        args and kwargs are put into the object store just once, saving time
        and memory. That method is then ran.

        If func is not a BatchedCalls instance, it is returned without changes.

        The ObjectRefs are cached inside two registries (_registry and
        _registry_hashable), which are common for the entire Pool and are
        cleaned on close."""
    if RayBatchedCalls is None:
        return func
    orginal_func = func
    if isinstance(func, SafeFunction):
        func = func.func
    if isinstance(func, BatchedCalls):
        func = RayBatchedCalls(func.items, (func._backend, func._n_jobs), func._reducer_callback, func._pickle_cache)
        func.put_items_in_object_store(self._registry, self._registry_hashable)
    else:
        func = orginal_func
    return func