from __future__ import annotations
import asyncio  # noqa
import typing
from typing import Any
from typing import Callable
from typing import Coroutine
from typing import TypeVar
def _util_async_run(fn, *arg, **kw):
    return fn(*arg, **kw)