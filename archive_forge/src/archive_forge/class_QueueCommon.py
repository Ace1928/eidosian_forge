from __future__ import annotations
import asyncio
from collections import deque
import threading
from time import time as _time
import typing
from typing import Any
from typing import Awaitable
from typing import Deque
from typing import Generic
from typing import Optional
from typing import TypeVar
from .concurrency import await_fallback
from .concurrency import await_only
from .langhelpers import memoized_property
class QueueCommon(Generic[_T]):
    maxsize: int
    use_lifo: bool

    def __init__(self, maxsize: int=0, use_lifo: bool=False):
        ...

    def empty(self) -> bool:
        raise NotImplementedError()

    def full(self) -> bool:
        raise NotImplementedError()

    def qsize(self) -> int:
        raise NotImplementedError()

    def put_nowait(self, item: _T) -> None:
        raise NotImplementedError()

    def put(self, item: _T, block: bool=True, timeout: Optional[float]=None) -> None:
        raise NotImplementedError()

    def get_nowait(self) -> _T:
        raise NotImplementedError()

    def get(self, block: bool=True, timeout: Optional[float]=None) -> _T:
        raise NotImplementedError()