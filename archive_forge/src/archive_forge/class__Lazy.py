from asyncio import AbstractEventLoop, new_event_loop, run_coroutine_threadsafe
from concurrent.futures import Future
from threading import Thread, Lock
from typing import ContextManager, Generic, TypeVar, Optional, Callable
class _Lazy(Generic[_T]):
    _Factory = Callable[[], _T]
    _lock: Lock
    _factory: _Factory
    _impl: Optional[_T]

    def __init__(self, factory: _Factory):
        self._lock = Lock()
        self._factory = factory
        self._impl = None

    def get(self) -> _T:
        with self._lock:
            if self._impl is None:
                self._impl = self._factory()
            return self._impl