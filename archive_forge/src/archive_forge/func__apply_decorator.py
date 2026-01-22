import os
from io import open
import types
from functools import wraps, partial
from contextlib import contextmanager
import sys, re
import sre_parse
import sre_constants
from inspect import getmembers, getmro
from functools import partial, wraps
from itertools import repeat, product
@classmethod
def _apply_decorator(cls, decorator, **kwargs):
    mro = getmro(cls)
    assert mro[0] is cls
    libmembers = {name for _cls in mro[1:] for name, _ in getmembers(_cls)}
    for name, value in getmembers(cls):
        if name.startswith('_') or (name in libmembers and name not in cls.__dict__):
            continue
        if not callable(cls.__dict__[name]):
            continue
        if hasattr(cls.__dict__[name], 'vargs_applied'):
            continue
        static = isinstance(cls.__dict__[name], (staticmethod, classmethod))
        setattr(cls, name, decorator(value, static=static, **kwargs))
    return cls