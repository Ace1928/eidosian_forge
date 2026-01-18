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
def try_catch(handler, *args, **kwargs):
    try:
        return (handler(*args, **kwargs), None)
    except Exception as e:
        tb = exc_info()[2]
        return (None, (e, tb))