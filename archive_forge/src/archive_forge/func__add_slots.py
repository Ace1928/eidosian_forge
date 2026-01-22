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
def _add_slots(cls, is_frozen, weakref_slot):
    if '__slots__' in cls.__dict__:
        raise TypeError(f'{cls.__name__} already specifies __slots__')
    cls_dict = dict(cls.__dict__)
    field_names = tuple((f.name for f in fields(cls)))
    inherited_slots = set(itertools.chain.from_iterable(map(_get_slots, cls.__mro__[1:-1])))
    cls_dict['__slots__'] = tuple(itertools.filterfalse(inherited_slots.__contains__, itertools.chain(field_names, ('__weakref__',) if weakref_slot else ())))
    for field_name in field_names:
        cls_dict.pop(field_name, None)
    cls_dict.pop('__dict__', None)
    cls_dict.pop('__weakref__', None)
    qualname = getattr(cls, '__qualname__', None)
    cls = type(cls)(cls.__name__, cls.__bases__, cls_dict)
    if qualname is not None:
        cls.__qualname__ = qualname
    if is_frozen:
        if '__getstate__' not in cls_dict:
            cls.__getstate__ = _dataclass_getstate
        if '__setstate__' not in cls_dict:
            cls.__setstate__ = _dataclass_setstate
    return cls