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
def _resolve_callback(self, value):
    if value is self:
        return self._reject_callback(make_self_resolution_error(), False)
    if not self.is_thenable(value):
        return self._fulfill(value)
    promise = self._try_convert_to_promise(value)._target()
    if promise == self:
        self._reject(make_self_resolution_error())
        return
    if promise._state == STATE_PENDING:
        len = self._length
        if len > 0:
            promise._migrate_callback0(self)
        for i in range(1, len):
            promise._migrate_callback_at(self, i)
        self._is_following = True
        self._length = 0
        self._set_followee(promise)
    elif promise._state == STATE_FULFILLED:
        self._fulfill(promise._value())
    elif promise._state == STATE_REJECTED:
        self._reject(promise._reason(), promise._target()._traceback)