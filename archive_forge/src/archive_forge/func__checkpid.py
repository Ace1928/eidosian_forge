import asyncio
import enum
import errno
import inspect
import io
import os
import socket
import ssl
import threading
import warnings
from distutils.version import StrictVersion
from itertools import chain
from types import MappingProxyType
from typing import (
from urllib.parse import ParseResult, parse_qs, unquote, urlparse
import async_timeout
from .compat import Protocol, TypedDict
from .exceptions import (
from .utils import str_if_bytes
def _checkpid(self):
    if self.pid != os.getpid():
        acquired = self._fork_lock.acquire(timeout=5)
        if not acquired:
            raise ChildDeadlockedError
        try:
            if self.pid != os.getpid():
                self.reset()
        finally:
            self._fork_lock.release()