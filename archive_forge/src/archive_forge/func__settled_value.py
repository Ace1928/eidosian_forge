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
def _settled_value(self, _raise=False):
    assert not self._is_following
    if self._state == STATE_FULFILLED:
        return self._rejection_handler0
    elif self._state == STATE_REJECTED:
        if _raise:
            raise_val = self._fulfillment_handler0
            raise raise_val.with_traceback(self._traceback)
        return self._fulfillment_handler0