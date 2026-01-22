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
@wraps(method)
def _invocation_actor_class_remote_span(self, args: Any=tuple(), kwargs: MutableMapping[Any, Any]=None, *_args: Any, **_kwargs: Any):
    if kwargs is None:
        kwargs = {}
    if not _is_tracing_enabled():
        assert '_ray_trace_ctx' not in kwargs
        return method(self, args, kwargs, *_args, **_kwargs)
    class_name = self.__ray_metadata__.class_name
    method_name = '__init__'
    assert '_ray_trace_ctx' not in _kwargs
    tracer = _opentelemetry.trace.get_tracer(__name__)
    with tracer.start_as_current_span(name=_actor_span_producer_name(class_name, method_name), kind=_opentelemetry.trace.SpanKind.PRODUCER, attributes=_actor_hydrate_span_args(class_name, method_name)) as span:
        kwargs['_ray_trace_ctx'] = _DictPropagator.inject_current_context()
        result = method(self, args, kwargs, *_args, **_kwargs)
        span.set_attribute('ray.actor_id', result._ray_actor_id.hex())
        return result