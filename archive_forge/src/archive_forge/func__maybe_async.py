from __future__ import annotations
from functools import wraps
import inspect
from . import config
from ..util.concurrency import _AsyncUtil
def _maybe_async(fn, *args, **kwargs):
    """Run a function in an asyncio loop if the current selected driver is
    async.

    This function is used for test setup/teardown and tests themselves
    where the current DB driver is known.


    """
    if not ENABLE_ASYNCIO:
        return fn(*args, **kwargs)
    is_async = config._current.is_async
    if is_async:
        return _async_util.run_in_greenlet(fn, *args, **kwargs)
    else:
        return fn(*args, **kwargs)