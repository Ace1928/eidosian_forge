import concurrent.futures
import contextvars
import functools
import inspect
import itertools
import types
import warnings
import weakref
from types import GenericAlias
from . import base_tasks
from . import coroutines
from . import events
from . import exceptions
from . import futures
from .coroutines import _is_coroutine
def _set_task_name(task, name):
    if name is not None:
        try:
            set_name = task.set_name
        except AttributeError:
            warnings.warn('Task.set_name() was added in Python 3.8, the method support will be mandatory for third-party task implementations since 3.13.', DeprecationWarning, stacklevel=3)
        else:
            set_name(name)