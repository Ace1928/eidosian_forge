import sys
from functools import wraps
from types import coroutine
import inspect
from inspect import (
import collections.abc
def unpack_StopAsyncIteration(e):
    if e.args:
        return e.args[0]
    else:
        return None