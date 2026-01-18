from __future__ import annotations
from functools import wraps
import inspect
from . import config
from ..util.concurrency import _AsyncUtil
@wraps(fn)
def wrap_fixture(*args, **kwargs):
    return _maybe_async(fn, *args, **kwargs)