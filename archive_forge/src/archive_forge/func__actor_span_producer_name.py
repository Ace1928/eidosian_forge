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
def _actor_span_producer_name(class_: Union[str, Callable[..., Any]], method: Union[str, Callable[..., Any]]) -> str:
    """Returns the actor span name that has span kind of producer."""
    if not isinstance(class_, str):
        class_ = class_.__name__
    if not isinstance(method, str):
        method = method.__name__
    return f'{class_}.{method} ray.remote'