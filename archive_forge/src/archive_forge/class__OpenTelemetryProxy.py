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
class _OpenTelemetryProxy:
    """
    This proxy makes it possible for tracing to be disabled when opentelemetry
    is not installed on the cluster, but is installed locally.

    The check for `opentelemetry`'s existence must happen where the functions
    are executed because `opentelemetry` may be present where the functions
    are pickled. This can happen when `ray[full]` is installed locally by `ray`
    (no extra dependencies) is installed on the cluster.
    """
    allowed_functions = {'trace', 'context', 'propagate', 'Context'}

    def __getattr__(self, name):
        if name in _OpenTelemetryProxy.allowed_functions:
            return getattr(self, f'_{name}')()
        else:
            raise AttributeError(f'Attribute does not exist: {name}')

    def _trace(self):
        return self._try_import('opentelemetry.trace')

    def _context(self):
        return self._try_import('opentelemetry.context')

    def _propagate(self):
        return self._try_import('opentelemetry.propagate')

    def _Context(self):
        context = self._context()
        if context:
            return context.context.Context
        else:
            return None

    def try_all(self):
        self._trace()
        self._context()
        self._propagate()
        self._Context()

    def _try_import(self, module):
        try:
            return importlib.import_module(module)
        except ImportError:
            if os.getenv('RAY_TRACING_ENABLED', 'False').lower() in ['true', '1']:
                raise ImportError("Install opentelemetry with 'pip install opentelemetry-api==1.0.0rc1' and 'pip install opentelemetry-sdk==1.0.0rc1' to enable tracing. See more at docs.ray.io/tracing.html")