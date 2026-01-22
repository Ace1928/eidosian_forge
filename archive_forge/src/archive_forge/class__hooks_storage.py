import sys
from functools import wraps
from types import coroutine
import inspect
from inspect import (
import collections.abc
class _hooks_storage(threading.local):

    def __init__(self):
        self.firstiter = None
        self.finalizer = None