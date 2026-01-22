import functools
import inspect
import logging
def _get_full_class_name(cls):
    return '%s.%s' % (cls.__module__, getattr(cls, '__qualname__', cls.__name__))