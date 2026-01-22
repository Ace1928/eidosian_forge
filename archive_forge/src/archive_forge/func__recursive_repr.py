import re
import sys
import copy
import types
import inspect
import keyword
import builtins
import functools
import itertools
import abc
import _thread
from types import FunctionType, GenericAlias
def _recursive_repr(user_function):
    repr_running = set()

    @functools.wraps(user_function)
    def wrapper(self):
        key = (id(self), _thread.get_ident())
        if key in repr_running:
            return '...'
        repr_running.add(key)
        try:
            result = user_function(self)
        finally:
            repr_running.discard(key)
        return result
    return wrapper