from __future__ import annotations
import collections.abc
import inspect
import os
import signal
import threading
from abc import ABCMeta
from functools import update_wrapper
from typing import (
from sniffio import thread_local as sniffio_loop
import trio
class generic_function(Generic[RetT]):
    """Decorator that makes a function indexable, to communicate
    non-inferrable generic type parameters to a static type checker.

    If you write::

        @generic_function
        def open_memory_channel(max_buffer_size: int) -> Tuple[
            SendChannel[T], ReceiveChannel[T]
        ]: ...

    it is valid at runtime to say ``open_memory_channel[bytes](5)``.
    This behaves identically to ``open_memory_channel(5)`` at runtime,
    and currently won't type-check without a mypy plugin or clever stubs,
    but at least it becomes possible to write those.
    """

    def __init__(self, fn: Callable[..., RetT]) -> None:
        update_wrapper(self, fn)
        self._fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> RetT:
        return self._fn(*args, **kwargs)

    def __getitem__(self, subscript: object) -> Self:
        return self