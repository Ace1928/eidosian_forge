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
@wraps(function)
def _function_with_tracing(*args: Any, _ray_trace_ctx: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Any:
    if _ray_trace_ctx is None:
        return function(*args, **kwargs)
    tracer = _opentelemetry.trace.get_tracer(__name__)
    function_name = function.__module__ + '.' + function.__name__
    with _use_context(_DictPropagator.extract(_ray_trace_ctx)), tracer.start_as_current_span(_function_span_consumer_name(function_name), kind=_opentelemetry.trace.SpanKind.CONSUMER, attributes=_function_hydrate_span_args(function_name)):
        return function(*args, **kwargs)