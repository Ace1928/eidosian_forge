import abc
import os
import sys
import _collections_abc
from collections import deque
from functools import wraps
from types import MethodType, GenericAlias
def _fix_exception_context(new_exc, old_exc):
    while 1:
        exc_context = new_exc.__context__
        if exc_context is None or exc_context is old_exc:
            return
        if exc_context is frame_exc:
            break
        new_exc = exc_context
    new_exc.__context__ = old_exc