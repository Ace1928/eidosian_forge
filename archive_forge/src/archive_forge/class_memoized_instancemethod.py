from ast import parse
import codecs
import collections
import operator
import os
import re
import timeit
from .compat import importlib_metadata_get
class memoized_instancemethod:
    """Decorate a method memoize its return value.

    Best applied to no-arg methods: memoization is not sensitive to
    argument values, and will always return the same value even when
    called with different arguments.

    """

    def __init__(self, fget, doc=None):
        self.fget = fget
        self.__doc__ = doc or fget.__doc__
        self.__name__ = fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return self

        def oneshot(*args, **kw):
            result = self.fget(obj, *args, **kw)

            def memo(*a, **kw):
                return result
            memo.__name__ = self.__name__
            memo.__doc__ = self.__doc__
            obj.__dict__[self.__name__] = memo
            return result
        oneshot.__name__ = self.__name__
        oneshot.__doc__ = self.__doc__
        return oneshot