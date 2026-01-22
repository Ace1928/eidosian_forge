import sys
from functools import partial
from inspect import isclass
from threading import Lock, RLock
from .arguments import formatargspec
from .__wrapt__ import (FunctionWrapper, BoundFunctionWrapper, ObjectProxy,
@decorator
def _synchronized(wrapped, instance, args, kwargs):
    with lock:
        return wrapped(*args, **kwargs)