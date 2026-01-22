from __future__ import annotations
import collections
import enum
from functools import update_wrapper
import inspect
import itertools
import operator
import re
import sys
import textwrap
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
from . import _collections
from . import compat
from ._has_cy import HAS_CYEXTENSION
from .typing import Literal
from .. import exc
class _IntFlagMeta(type):

    def __init__(cls, classname: str, bases: Tuple[Type[Any], ...], dict_: Dict[str, Any], **kw: Any) -> None:
        items: List[symbol]
        cls._items = items = []
        for k, v in dict_.items():
            if isinstance(v, int):
                sym = symbol(k, canonical=v)
            elif not k.startswith('_'):
                raise TypeError('Expected integer values for IntFlag')
            else:
                continue
            setattr(cls, k, sym)
            items.append(sym)
        cls.__members__ = _collections.immutabledict({sym.name: sym for sym in items})

    def __iter__(self) -> Iterator[symbol]:
        raise NotImplementedError('iter not implemented to ensure compatibility with Python 3.11 IntFlag.  Please use __members__.  See https://github.com/python/cpython/issues/99304')