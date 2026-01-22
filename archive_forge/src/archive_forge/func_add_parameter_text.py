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
def add_parameter_text(params: Any, text: str) -> Callable[[_F], _F]:
    params = _collections.to_list(params)

    def decorate(fn):
        doc = fn.__doc__ is not None and fn.__doc__ or ''
        if doc:
            doc = inject_param_text(doc, {param: text for param in params})
        fn.__doc__ = doc
        return fn
    return decorate