import inspect
import itertools
import os
import warnings
from functools import partial
from importlib.metadata import entry_points
import networkx as nx
from .decorators import argmap
from .configs import Config, config
def _should_backend_run(self, backend_name, /, *args, **kwargs):
    """Can/should the specified backend run this algorithm with these arguments?"""
    backend = _load_backend(backend_name)
    return hasattr(backend, self.name) and (can_run := backend.can_run(self.name, args, kwargs)) and (not isinstance(can_run, str)) and (should_run := backend.should_run(self.name, args, kwargs)) and (not isinstance(should_run, str))