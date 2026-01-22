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
class _RawSocketMixin:
    _receive_future: asyncio.Future | None = None
    _send_future: asyncio.Future | None = None
    _closing = False

    def __init__(self, raw_socket: socket.socket):
        self.__raw_socket = raw_socket
        self._receive_guard = ResourceGuard('reading from')
        self._send_guard = ResourceGuard('writing to')

    @property
    def _raw_socket(self) -> socket.socket:
        return self.__raw_socket

    def _wait_until_readable(self, loop: asyncio.AbstractEventLoop) -> asyncio.Future:

        def callback(f: object) -> None:
            del self._receive_future
            loop.remove_reader(self.__raw_socket)
        f = self._receive_future = asyncio.Future()
        loop.add_reader(self.__raw_socket, f.set_result, None)
        f.add_done_callback(callback)
        return f

    def _wait_until_writable(self, loop: asyncio.AbstractEventLoop) -> asyncio.Future:

        def callback(f: object) -> None:
            del self._send_future
            loop.remove_writer(self.__raw_socket)
        f = self._send_future = asyncio.Future()
        loop.add_writer(self.__raw_socket, f.set_result, None)
        f.add_done_callback(callback)
        return f

    async def aclose(self) -> None:
        if not self._closing:
            self._closing = True
            if self.__raw_socket.fileno() != -1:
                self.__raw_socket.close()
            if self._receive_future:
                self._receive_future.set_result(None)
            if self._send_future:
                self._send_future.set_result(None)