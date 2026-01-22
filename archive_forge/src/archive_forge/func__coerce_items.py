from __future__ import annotations as _annotations
import dataclasses
import keyword
import typing
import weakref
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from itertools import zip_longest
from types import BuiltinFunctionType, CodeType, FunctionType, GeneratorType, LambdaType, ModuleType
from typing import Any, Mapping, TypeVar
from typing_extensions import TypeAlias, TypeGuard
from . import _repr, _typing_extra
@staticmethod
def _coerce_items(items: AbstractSetIntStr | MappingIntStrAny) -> MappingIntStrAny:
    if isinstance(items, typing.Mapping):
        pass
    elif isinstance(items, typing.AbstractSet):
        items = dict.fromkeys(items, ...)
    else:
        class_name = getattr(items, '__class__', '???')
        raise TypeError(f'Unexpected type of exclude value {class_name}')
    return items