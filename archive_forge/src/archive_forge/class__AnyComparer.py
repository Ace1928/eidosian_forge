import asyncio
import contextlib
import io
import inspect
import pprint
import sys
import builtins
import pkgutil
from asyncio import iscoroutinefunction
from types import CodeType, ModuleType, MethodType
from unittest.util import safe_repr
from functools import wraps, partial
from threading import RLock
class _AnyComparer(list):
    """A list which checks if it contains a call which may have an
    argument of ANY, flipping the components of item and self from
    their traditional locations so that ANY is guaranteed to be on
    the left."""

    def __contains__(self, item):
        for _call in self:
            assert len(item) == len(_call)
            if all([expected == actual for expected, actual in zip(item, _call)]):
                return True
        return False