import asyncio
import base64
import binascii
import contextlib
import datetime
import enum
import functools
import inspect
import netrc
import os
import platform
import re
import sys
import time
import warnings
import weakref
from collections import namedtuple
from contextlib import suppress
from email.parser import HeaderParser
from email.utils import parsedate
from math import ceil
from pathlib import Path
from types import TracebackType
from typing import (
from urllib.parse import quote
from urllib.request import getproxies, proxy_bypass
import attr
from multidict import MultiDict, MultiDictProxy, MultiMapping
from yarl import URL
from . import hdrs
from .log import client_logger, internal_logger
class TimeoutHandle:
    """Timeout handle"""

    def __init__(self, loop: asyncio.AbstractEventLoop, timeout: Optional[float], ceil_threshold: float=5) -> None:
        self._timeout = timeout
        self._loop = loop
        self._ceil_threshold = ceil_threshold
        self._callbacks: List[Tuple[Callable[..., None], Tuple[Any, ...], Dict[str, Any]]] = []

    def register(self, callback: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        self._callbacks.append((callback, args, kwargs))

    def close(self) -> None:
        self._callbacks.clear()

    def start(self) -> Optional[asyncio.Handle]:
        timeout = self._timeout
        if timeout is not None and timeout > 0:
            when = self._loop.time() + timeout
            if timeout >= self._ceil_threshold:
                when = ceil(when)
            return self._loop.call_at(when, self.__call__)
        else:
            return None

    def timer(self) -> 'BaseTimerContext':
        if self._timeout is not None and self._timeout > 0:
            timer = TimerContext(self._loop)
            self.register(timer.timeout)
            return timer
        else:
            return TimerNoop()

    def __call__(self) -> None:
        for cb, args, kwargs in self._callbacks:
            with suppress(Exception):
                cb(*args, **kwargs)
        self._callbacks.clear()