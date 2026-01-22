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
class EnsureKWArg:
    """Apply translation of functions to accept \\**kw arguments if they
    don't already.

    Used to ensure cross-compatibility with third party legacy code, for things
    like compiler visit methods that need to accept ``**kw`` arguments,
    but may have been copied from old code that didn't accept them.

    """
    ensure_kwarg: str
    'a regular expression that indicates method names for which the method\n    should accept ``**kw`` arguments.\n\n    The class will scan for methods matching the name template and decorate\n    them if necessary to ensure ``**kw`` parameters are accepted.\n\n    '

    def __init_subclass__(cls) -> None:
        fn_reg = cls.ensure_kwarg
        clsdict = cls.__dict__
        if fn_reg:
            for key in clsdict:
                m = re.match(fn_reg, key)
                if m:
                    fn = clsdict[key]
                    spec = compat.inspect_getfullargspec(fn)
                    if not spec.varkw:
                        wrapped = cls._wrap_w_kw(fn)
                        setattr(cls, key, wrapped)
        super().__init_subclass__()

    @classmethod
    def _wrap_w_kw(cls, fn: Callable[..., Any]) -> Callable[..., Any]:

        def wrap(*arg: Any, **kw: Any) -> Any:
            return fn(*arg)
        return update_wrapper(wrap, fn)