import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
@functools.lru_cache(None)
def _infer_class_backend_cached(cls):
    try:
        import numpy as _numpy
        if issubclass(cls, _numpy.ndarray):
            return 'numpy'
    except ImportError:
        pass
    if cls in _CUSTOM_BACKENDS:
        return _CUSTOM_BACKENDS[cls]
    lib = cls.__module__.split('.')[0]
    backend = _BACKEND_ALIASES.get(lib, lib)
    return backend