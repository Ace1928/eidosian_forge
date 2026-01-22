import threading
from types import TracebackType
from typing import Optional, Type
from ._exceptions import ExceptionMapping, PoolTimeout, map_exceptions
class AsyncShieldCancellation:

    def __init__(self) -> None:
        """
        Detect if we're running under 'asyncio' or 'trio' and create
        a shielded scope with the correct implementation.
        """
        self._backend = current_async_library()
        if self._backend == 'trio':
            self._trio_shield = trio.CancelScope(shield=True)
        elif self._backend == 'asyncio':
            self._anyio_shield = anyio.CancelScope(shield=True)

    def __enter__(self) -> 'AsyncShieldCancellation':
        if self._backend == 'trio':
            self._trio_shield.__enter__()
        elif self._backend == 'asyncio':
            self._anyio_shield.__enter__()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]]=None, exc_value: Optional[BaseException]=None, traceback: Optional[TracebackType]=None) -> None:
        if self._backend == 'trio':
            self._trio_shield.__exit__(exc_type, exc_value, traceback)
        elif self._backend == 'asyncio':
            self._anyio_shield.__exit__(exc_type, exc_value, traceback)