from __future__ import annotations
from .. import mesonlib, mlog
from .disabler import Disabler
from .exceptions import InterpreterException, InvalidArguments
from ._unholder import _unholder
from dataclasses import dataclass
from functools import wraps
import abc
import itertools
import copy
import typing as T
def types_description(types_tuple: T.Tuple[T.Union[T.Type, ContainerTypeInfo], ...]) -> str:
    candidates = []
    for t in types_tuple:
        if isinstance(t, ContainerTypeInfo):
            candidates.append(t.description())
        else:
            candidates.append(t.__name__)
    shouldbe = 'one of: ' if len(candidates) > 1 else ''
    shouldbe += ', '.join(candidates)
    return shouldbe