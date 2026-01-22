from collections import namedtuple
from functools import partial, wraps
from sys import version_info, exc_info
from threading import RLock
from types import TracebackType
from weakref import WeakKeyDictionary
from .async_ import Async
from .compat import (
from .utils import deprecated, integer_types, string_types, text_type, binary_type, warn
from .promise_list import PromiseList
from .schedulers.immediate import ImmediateScheduler
from typing import TypeVar, Generic
def _settle_promise(self, promise, handler, value, traceback):
    assert not self._is_following
    is_promise = isinstance(promise, self.__class__)
    async_guaranteed = self._is_async_guaranteed
    if callable(handler):
        if not is_promise:
            handler(value)
        else:
            if async_guaranteed:
                promise._is_async_guaranteed = True
            self._settle_promise_from_handler(handler, value, promise)
    elif is_promise:
        if async_guaranteed:
            promise._is_async_guaranteed = True
        if self._state == STATE_FULFILLED:
            promise._fulfill(value)
        else:
            promise._reject(value, self._traceback)