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
def _fulfill(self, value):
    if value is self:
        err = make_self_resolution_error()
        return self._reject(err)
    self._state = STATE_FULFILLED
    self._rejection_handler0 = value
    if self._length > 0:
        if self._is_async_guaranteed:
            self._settle_promises()
        else:
            async_instance.settle_promises(self)