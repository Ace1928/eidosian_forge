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
def _iter_side_effect():
    if handle.readline.return_value is not None:
        while True:
            yield handle.readline.return_value
    for line in _state[0]:
        yield line