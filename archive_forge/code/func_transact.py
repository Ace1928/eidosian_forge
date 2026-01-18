from __future__ import annotations
import os
import io
import abc
import zlib
import errno
import time
import struct
import sqlite3
import threading
import pickletools
import asyncio
import inspect
import dill as pkl
import functools as ft
import contextlib as cl
import warnings
from fileio.lib.types import File, FileLike
from typing import Any, Optional, Type, Dict, Union, Tuple, TYPE_CHECKING
from lazyops.utils.pooler import ThreadPooler
from lazyops.libs.sqlcache.constants import (
from lazyops.libs.sqlcache.config import SqlCacheConfig
from lazyops.libs.sqlcache.exceptions import (
from lazyops.libs.sqlcache.utils import (
@cl.contextmanager
def transact(self, retry=False):
    """Context manager to perform a transaction by locking the cache.
        While the cache is locked, no other write operation is permitted.
        Transactions should therefore be as short as possible. Read and write
        operations performed in a transaction are atomic. Read operations may
        occur concurrent to a transaction.
        Transactions may be nested and may not be shared between threads.
        Raises :exc:`Timeout` error when database timeout occurs and `retry` is
        `False` (default).
        >>> cache = Store()
        >>> with cache.transact():  # Atomically increment two keys.
        ...     _ = cache.incr('total', 123.4)
        ...     _ = cache.incr('count', 1)
        >>> with cache.transact():  # Atomically calculate average.
        ...     average = cache['total'] / cache['count']
        >>> average
        123.4
        :param bool retry: retry if database timeout occurs (default False)
        :return: context manager for use in `with` statement
        :raises Timeout: if database timeout occurs
        """
    with self._transact(retry=retry):
        yield