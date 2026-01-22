from __future__ import annotations
import asyncio
import functools
import logging
import sys
import typing
from .abstract_loop import EventLoop, ExitMainLoop
def _also_call_idle(self, callback: Callable[_Spec, _T]) -> Callable[_Spec, _T]:
    """
        Wrap the callback to also call _entering_idle.
        """

    @functools.wraps(callback)
    def wrapper(*args: _Spec.args, **kwargs: _Spec.kwargs) -> _T:
        if not self._idle_asyncio_handle:
            self._idle_asyncio_handle = self._loop.call_later(0, self._entering_idle)
        return callback(*args, **kwargs)
    return wrapper