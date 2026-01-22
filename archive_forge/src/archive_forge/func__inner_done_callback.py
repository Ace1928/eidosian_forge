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
def _inner_done_callback(inner):
    if outer.cancelled():
        if not inner.cancelled():
            inner.exception()
        return
    if inner.cancelled():
        outer.cancel()
    else:
        exc = inner.exception()
        if exc is not None:
            outer.set_exception(exc)
        else:
            outer.set_result(inner.result())