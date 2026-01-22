from __future__ import annotations
from functools import wraps
import inspect
from . import config
from ..util.concurrency import _AsyncUtil
def _assume_async(fn, *args, **kwargs):
    """Run a function in an asyncio loop unconditionally.

    This function is used for provisioning features like
    testing a database connection for server info.

    Note that for blocking IO database drivers, this means they block the
    event loop.

    """
    if not ENABLE_ASYNCIO:
        return fn(*args, **kwargs)
    return _async_util.run_in_greenlet(fn, *args, **kwargs)