from __future__ import annotations
import asyncio
import functools
import logging
import sys
import typing
from .abstract_loop import EventLoop, ExitMainLoop
def _entering_idle(self) -> None:
    """
        Call all the registered idle callbacks.
        """
    try:
        for callback in self._idle_callbacks.values():
            callback()
    finally:
        self._idle_asyncio_handle = None