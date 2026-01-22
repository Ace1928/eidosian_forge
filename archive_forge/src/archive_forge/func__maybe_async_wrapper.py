from __future__ import annotations
from functools import wraps
import inspect
from . import config
from ..util.concurrency import _AsyncUtil
def _maybe_async_wrapper(fn):
    """Apply the _maybe_async function to an existing function and return
    as a wrapped callable, supporting generator functions as well.

    This is currently used for pytest fixtures that support generator use.

    """
    if inspect.isgeneratorfunction(fn):
        _stop = object()

        def call_next(gen):
            try:
                return next(gen)
            except StopIteration:
                return _stop

        @wraps(fn)
        def wrap_fixture(*args, **kwargs):
            gen = fn(*args, **kwargs)
            while True:
                value = _maybe_async(call_next, gen)
                if value is _stop:
                    break
                yield value
    else:

        @wraps(fn)
        def wrap_fixture(*args, **kwargs):
            return _maybe_async(fn, *args, **kwargs)
    return wrap_fixture