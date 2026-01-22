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
def _return_value_looks_like_wrong_library(value: object) -> bool:
    if isinstance(value, collections.abc.Generator):
        return True
    if getattr(value, '_asyncio_future_blocking', None) is not None:
        return True
    if value.__class__.__name__ in ('Future', 'Deferred'):
        return True
    return False