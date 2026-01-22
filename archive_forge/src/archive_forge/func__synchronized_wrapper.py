import sys
from functools import partial
from inspect import isclass
from threading import Lock, RLock
from .arguments import formatargspec
from .__wrapt__ import (FunctionWrapper, BoundFunctionWrapper, ObjectProxy,
def _synchronized_wrapper(wrapped, instance, args, kwargs):
    with _synchronized_lock(instance if instance is not None else wrapped):
        return wrapped(*args, **kwargs)