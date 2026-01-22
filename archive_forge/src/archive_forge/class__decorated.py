from __future__ import annotations
import os
from functools import wraps
from typing import Hashable, TypeVar
@wraps(klass, assigned=('__name__', '__module__'), updated=())
class _decorated(klass):
    __doc__ = klass.__doc__

    def __new__(cls, *args, **kwargs):
        """
            Pass through.
            """
        key = (cls, *args, *tuple(kwargs.items()))
        try:
            inst = cache.get(key)
        except TypeError:
            inst = key = None
        if inst is None:
            inst = klass(*args, **kwargs)
            inst.__class__ = cls
            if key is not None:
                cache[key] = inst
        return inst

    def __init__(self, *args, **kwargs):
        pass