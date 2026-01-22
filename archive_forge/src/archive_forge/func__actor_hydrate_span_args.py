import importlib
import inspect
import logging
import os
from contextlib import contextmanager
from functools import wraps
from inspect import Parameter
from types import ModuleType
from typing import (
import ray
import ray._private.worker
from ray._private.inspect_util import (
from ray.runtime_context import get_runtime_context
def _actor_hydrate_span_args(class_: Union[str, Callable[..., Any]], method: Union[str, Callable[..., Any]]):
    """Get the Attributes of the actor that will be reported as attributes
    in the trace."""
    if callable(class_):
        class_ = class_.__name__
    if callable(method):
        method = method.__name__
    runtime_context = get_runtime_context()
    span_args = {'ray.remote': 'actor', 'ray.actor_class': class_, 'ray.actor_method': method, 'ray.function': f'{class_}.{method}', 'ray.pid': str(os.getpid()), 'ray.job_id': runtime_context.get_job_id(), 'ray.node_id': runtime_context.get_node_id()}
    if ray._private.worker.global_worker.mode == ray._private.worker.WORKER_MODE:
        actor_id = runtime_context.get_actor_id()
        if actor_id:
            span_args['ray.actor_id'] = actor_id
    worker_id = getattr(ray._private.worker.global_worker, 'worker_id', None)
    if worker_id:
        span_args['ray.worker_id'] = worker_id.hex()
    return span_args