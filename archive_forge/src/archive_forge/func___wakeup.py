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
def __wakeup(self, future):
    try:
        future.result()
    except BaseException as exc:
        self.__step(exc)
    else:
        self.__step()
    self = None