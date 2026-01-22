from __future__ import annotations
import array
import asyncio
import concurrent.futures
import math
import socket
import sys
import threading
from asyncio import (
from asyncio.base_events import _run_until_complete_cb  # type: ignore[attr-defined]
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Generator, Iterable
from concurrent.futures import Future
from contextlib import suppress
from contextvars import Context, copy_context
from dataclasses import dataclass
from functools import partial, wraps
from inspect import (
from io import IOBase
from os import PathLike
from queue import Queue
from signal import Signals
from socket import AddressFamily, SocketKind
from threading import Thread
from types import TracebackType
from typing import (
from weakref import WeakKeyDictionary
import sniffio
from .. import CapacityLimiterStatistics, EventStatistics, TaskInfo, abc
from .._core._eventloop import claim_worker_thread, threadlocals
from .._core._exceptions import (
from .._core._sockets import convert_ipv6_sockaddr
from .._core._streams import create_memory_object_stream
from .._core._synchronization import CapacityLimiter as BaseCapacityLimiter
from .._core._synchronization import Event as BaseEvent
from .._core._synchronization import ResourceGuard
from .._core._tasks import CancelScope as BaseCancelScope
from ..abc import (
from ..lowlevel import RunVar
from ..streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
def _deliver_cancellation(self, origin: CancelScope) -> bool:
    """
        Deliver cancellation to directly contained tasks and nested cancel scopes.

        Schedule another run at the end if we still have tasks eligible for
        cancellation.

        :param origin: the cancel scope that originated the cancellation
        :return: ``True`` if the delivery needs to be retried on the next cycle

        """
    should_retry = False
    current = current_task()
    for task in self._tasks:
        if task._must_cancel:
            continue
        should_retry = True
        if task is not current and (task is self._host_task or _task_started(task)):
            waiter = task._fut_waiter
            if not isinstance(waiter, asyncio.Future) or not waiter.done():
                self._cancel_calls += 1
                if sys.version_info >= (3, 9):
                    task.cancel(f'Cancelled by cancel scope {id(origin):x}')
                else:
                    task.cancel()
    for scope in self._child_scopes:
        if not scope._shield and (not scope.cancel_called):
            should_retry = scope._deliver_cancellation(origin) or should_retry
    if origin is self:
        if should_retry:
            self._cancel_handle = get_running_loop().call_soon(self._deliver_cancellation, origin)
        else:
            self._cancel_handle = None
    return should_retry