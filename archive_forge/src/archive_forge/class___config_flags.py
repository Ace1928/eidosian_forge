import inspect
import warnings
import types
import collections
import itertools
from functools import lru_cache, wraps
from typing import Callable, List, Union, Iterable, TypeVar, cast
class __config_flags:
    """Internal class for defining compatibility and debugging flags"""
    _all_names: List[str] = []
    _fixed_names: List[str] = []
    _type_desc = 'configuration'

    @classmethod
    def _set(cls, dname, value):
        if dname in cls._fixed_names:
            warnings.warn(f'{cls.__name__}.{dname} {cls._type_desc} is {str(getattr(cls, dname)).upper()} and cannot be overridden', stacklevel=3)
            return
        if dname in cls._all_names:
            setattr(cls, dname, value)
        else:
            raise ValueError(f'no such {cls._type_desc} {dname!r}')
    enable = classmethod(lambda cls, name: cls._set(name, True))
    disable = classmethod(lambda cls, name: cls._set(name, False))