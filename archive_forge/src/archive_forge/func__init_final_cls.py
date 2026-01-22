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
def _init_final_cls(cls: type[object]) -> NoReturn:
    """Raises an exception when a final class is subclassed."""
    raise TypeError(f'{cls.__module__}.{cls.__qualname__} does not support subclassing')