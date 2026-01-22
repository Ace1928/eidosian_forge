from asyncio import AbstractEventLoop, new_event_loop, run_coroutine_threadsafe
from concurrent.futures import Future
from threading import Thread, Lock
from typing import ContextManager, Generic, TypeVar, Optional, Callable
class ManagedEventLoop(ContextManager):
    _loop: _ManagedEventLoopImpl

    def __init__(self, name=None):
        self._loop = _global_event_loop.get()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def submit(self, coro) -> Future:
        return self._loop.submit(coro)