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
def _function_span_producer_name(func: Callable[..., Any]) -> str:
    """Returns the function span name that has span kind of producer."""
    return f'{func} ray.remote'