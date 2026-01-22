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
class AsyncAdaptedQueue(QueueCommon[_T]):
    if typing.TYPE_CHECKING:

        @staticmethod
        def await_(coroutine: Awaitable[Any]) -> _T:
            ...
    else:
        await_ = staticmethod(await_only)

    def __init__(self, maxsize: int=0, use_lifo: bool=False):
        self.use_lifo = use_lifo
        self.maxsize = maxsize

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self):
        return self._queue.full()

    def qsize(self):
        return self._queue.qsize()

    @memoized_property
    def _queue(self) -> asyncio.Queue[_T]:
        queue: asyncio.Queue[_T]
        if self.use_lifo:
            queue = asyncio.LifoQueue(maxsize=self.maxsize)
        else:
            queue = asyncio.Queue(maxsize=self.maxsize)
        return queue

    def put_nowait(self, item: _T) -> None:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as err:
            raise Full() from err

    def put(self, item: _T, block: bool=True, timeout: Optional[float]=None) -> None:
        if not block:
            return self.put_nowait(item)
        try:
            if timeout is not None:
                self.await_(asyncio.wait_for(self._queue.put(item), timeout))
            else:
                self.await_(self._queue.put(item))
        except (asyncio.QueueFull, asyncio.TimeoutError) as err:
            raise Full() from err

    def get_nowait(self) -> _T:
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty as err:
            raise Empty() from err

    def get(self, block: bool=True, timeout: Optional[float]=None) -> _T:
        if not block:
            return self.get_nowait()
        try:
            if timeout is not None:
                return self.await_(asyncio.wait_for(self._queue.get(), timeout))
            else:
                return self.await_(self._queue.get())
        except (asyncio.QueueEmpty, asyncio.TimeoutError) as err:
            raise Empty() from err