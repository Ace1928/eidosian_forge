import inspect
import os
import sys
import warnings
from functools import partial
from importlib.metadata import entry_points
from ..exception import NetworkXNotImplemented
def _can_backend_run(self, backend_name, /, *args, **kwargs):
    """Can the specified backend run this algorithms with these arguments?"""
    backend = _load_backend(backend_name)
    return hasattr(backend, self.name) and (not hasattr(backend, 'can_run') or backend.can_run(self.name, args, kwargs))