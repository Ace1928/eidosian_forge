from __future__ import annotations
import sys
import threading
from collections.abc import Awaitable, Callable, Generator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, contextmanager
from inspect import isawaitable
from types import TracebackType
from typing import (
from ._core import _eventloop
from ._core._eventloop import get_async_backend, get_cancelled_exc_class, threadlocals
from ._core._synchronization import Event
from ._core._tasks import CancelScope, create_task_group
from .abc import AsyncBackend
from .abc._tasks import TaskStatus
def wrap_async_context_manager(self, cm: AsyncContextManager[T_co]) -> ContextManager[T_co]:
    """
        Wrap an async context manager as a synchronous context manager via this portal.

        Spawns a task that will call both ``__aenter__()`` and ``__aexit__()``, stopping
        in the middle until the synchronous context manager exits.

        :param cm: an asynchronous context manager
        :return: a synchronous context manager

        .. versionadded:: 2.1

        """
    return _BlockingAsyncContextManager(cm, self)