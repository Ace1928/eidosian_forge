import abc
import os
import sys
import _collections_abc
from collections import deque
from functools import wraps
from types import MethodType, GenericAlias
@staticmethod
def _create_cb_wrapper(callback, /, *args, **kwds):

    def _exit_wrapper(exc_type, exc, tb):
        callback(*args, **kwds)
    return _exit_wrapper