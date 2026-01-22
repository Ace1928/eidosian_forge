import types
import sys
import numbers
import functools
import copy
import inspect
def as_native_str(encoding='utf-8'):
    """
    A decorator to turn a function or method call that returns text, i.e.
    unicode, into one that returns a native platform str.

    Use it as a decorator like this::

        from __future__ import unicode_literals

        class MyClass(object):
            @as_native_str(encoding='ascii')
            def __repr__(self):
                return next(self._iter).upper()
    """
    if PY3:
        return lambda f: f
    else:

        def encoder(f):

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs).encode(encoding=encoding)
            return wrapper
        return encoder