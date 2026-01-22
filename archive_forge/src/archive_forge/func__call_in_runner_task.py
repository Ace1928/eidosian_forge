from __future__ import annotations
import array
import math
import socket
import sys
import types
from collections.abc import AsyncIterator, Iterable
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial
from io import IOBase
from os import PathLike
from signal import Signals
from socket import AddressFamily, SocketKind
from types import TracebackType
from typing import (
import trio.from_thread
import trio.lowlevel
from outcome import Error, Outcome, Value
from trio.lowlevel import (
from trio.socket import SocketType as TrioSocketType
from trio.to_thread import run_sync
from .. import CapacityLimiterStatistics, EventStatistics, TaskInfo, abc
from .._core._eventloop import claim_worker_thread
from .._core._exceptions import (
from .._core._sockets import convert_ipv6_sockaddr
from .._core._streams import create_memory_object_stream
from .._core._synchronization import CapacityLimiter as BaseCapacityLimiter
from .._core._synchronization import Event as BaseEvent
from .._core._synchronization import ResourceGuard
from .._core._tasks import CancelScope as BaseCancelScope
from ..abc import IPSockAddrType, UDPPacketType, UNIXDatagramPacketType
from ..abc._eventloop import AsyncBackend
from ..streams.memory import MemoryObjectSendStream
def _call_in_runner_task(self, func: Callable[P, Awaitable[T_Retval]], *args: P.args, **kwargs: P.kwargs) -> T_Retval:
    if self._send_stream is None:
        trio.lowlevel.start_guest_run(self._run_tests_and_fixtures, run_sync_soon_threadsafe=self._call_queue.put, done_callback=self._main_task_finished, **self._options)
        while self._send_stream is None:
            self._call_queue.get()()
    outcome_holder: list[Outcome] = []
    self._send_stream.send_nowait((func(*args, **kwargs), outcome_holder))
    while not outcome_holder:
        self._call_queue.get()()
    return outcome_holder[0].unwrap()