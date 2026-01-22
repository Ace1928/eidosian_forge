import sys
from functools import wraps
from types import coroutine
import inspect
from inspect import (
import collections.abc
@coroutine
def _yield_(value):
    return (yield _wrap(value))