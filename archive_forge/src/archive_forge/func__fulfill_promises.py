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
def _fulfill_promises(self, length, value):
    for i in range(1, length):
        handler = self._fulfillment_handler_at(i)
        promise = self._promise_at(i)
        self._clear_callback_data_index_at(i)
        self._settle_promise(promise, handler, value, None)