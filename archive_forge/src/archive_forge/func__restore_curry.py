from functools import reduce, partial
import inspect
import sys
from operator import attrgetter, not_
from importlib import import_module
from types import MethodType
from .utils import no_default
from . import _signatures as _sigs
def _restore_curry(cls, func, args, kwargs, userdict, is_decorated):
    if isinstance(func, str):
        modname, qualname = func.rsplit(':', 1)
        obj = import_module(modname)
        for attr in qualname.split('.'):
            obj = getattr(obj, attr)
        if is_decorated:
            return obj
        func = obj.func
    obj = cls(func, *args, **kwargs or {})
    obj.__dict__.update(userdict)
    return obj