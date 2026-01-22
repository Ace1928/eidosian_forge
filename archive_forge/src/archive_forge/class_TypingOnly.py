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
class TypingOnly:
    """A mixin class that marks a class as 'typing only', meaning it has
    absolutely no methods, attributes, or runtime functionality whatsoever.

    """
    __slots__ = ()

    def __init_subclass__(cls) -> None:
        if TypingOnly in cls.__bases__:
            remaining = set(cls.__dict__).difference({'__module__', '__doc__', '__slots__', '__orig_bases__', '__annotations__'})
            if remaining:
                raise AssertionError(f'Class {cls} directly inherits TypingOnly but has additional attributes {remaining}.')
        super().__init_subclass__()