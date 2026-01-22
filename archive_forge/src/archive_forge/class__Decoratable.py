from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
class _Decoratable:
    """Provides support for decorating methods with @v_args"""

    @classmethod
    def _apply_v_args(cls, visit_wrapper):
        mro = getmro(cls)
        assert mro[0] is cls
        libmembers = {name for _cls in mro[1:] for name, _ in getmembers(_cls)}
        for name, value in getmembers(cls):
            if name.startswith('_') or (name in libmembers and name not in cls.__dict__):
                continue
            if not callable(value):
                continue
            if isinstance(cls.__dict__[name], _VArgsWrapper):
                continue
            setattr(cls, name, _VArgsWrapper(cls.__dict__[name], visit_wrapper))
        return cls

    def __class_getitem__(cls, _):
        return cls