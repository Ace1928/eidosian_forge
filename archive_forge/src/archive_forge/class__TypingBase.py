import abc
from abc import abstractmethod, abstractproperty
import collections
import contextlib
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
class _TypingBase(metaclass=TypingMeta, _root=True):
    """Internal indicator of special typing constructs."""
    __slots__ = ('__weakref__',)

    def __init__(self, *args, **kwds):
        pass

    def __new__(cls, *args, **kwds):
        """Constructor.

        This only exists to give a better error message in case
        someone tries to subclass a special typing object (not a good idea).
        """
        if len(args) == 3 and isinstance(args[0], str) and isinstance(args[1], tuple):
            raise TypeError('Cannot subclass %r' % cls)
        return super().__new__(cls)

    def _eval_type(self, globalns, localns):
        return self

    def _get_type_vars(self, tvars):
        pass

    def __repr__(self):
        cls = type(self)
        qname = _trim_name(_qualname(cls))
        return '%s.%s' % (cls.__module__, qname)

    def __call__(self, *args, **kwds):
        raise TypeError('Cannot instantiate %r' % type(self))