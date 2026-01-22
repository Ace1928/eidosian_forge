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
@classmethod
def _try_convert_to_promise(cls, obj):
    _type = obj.__class__
    if issubclass(_type, Promise):
        if cls is not Promise:
            return cls(obj.then, obj._scheduler)
        return obj
    if iscoroutine(obj):
        obj = ensure_future(obj)
        _type = obj.__class__
    if is_future_like(_type):

        def executor(resolve, reject):
            if obj.done():
                _process_future_result(resolve, reject)(obj)
            else:
                obj.add_done_callback(_process_future_result(resolve, reject))
        promise = cls(executor)
        promise._future = obj
        return promise
    return obj