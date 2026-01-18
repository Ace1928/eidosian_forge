from __future__ import annotations
import time
import uuid
import typing
import random
import inspect
import functools
import datetime
import itertools
import asyncio
import contextlib
import async_lru
import signal
from pathlib import Path
from frozendict import frozendict
from typing import Dict, Callable, List, Any, Union, Coroutine, TypeVar, Optional, TYPE_CHECKING
from lazyops.utils.logs import default_logger
from lazyops.utils.serialization import (
from lazyops.utils.lazy import (
def with_fail_after(delay: float):
    """
    Creates a timeout for a function
    """

    def wrapper(func):

        @functools.wraps(func)
        def time_limited(*args, **kwargs):

            def handler(signum, frame):
                raise TimeoutError(f"Timeout for function '{func.__name__}'")
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(delay)
            result = None
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                raise exc
            finally:
                signal.alarm(0)
            return result
        return time_limited
    return wrapper