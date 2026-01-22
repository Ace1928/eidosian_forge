import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
def _ensure_subclassable(mro_entries):

    def inner(func):
        if sys.implementation.name == 'pypy' and sys.version_info < (3, 9):
            cls_dict = {'__call__': staticmethod(func), '__mro_entries__': staticmethod(mro_entries)}
            t = type(func.__name__, (), cls_dict)
            return functools.update_wrapper(t(), func)
        else:
            func.__mro_entries__ = mro_entries
            return func
    return inner