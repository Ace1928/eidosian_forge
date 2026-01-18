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
def wrap_callable(wrapper, fn):
    """Augment functools.update_wrapper() to work with objects with
    a ``__call__()`` method.

    :param fn:
      object with __call__ method

    """
    if hasattr(fn, '__name__'):
        return update_wrapper(wrapper, fn)
    else:
        _f = wrapper
        _f.__name__ = fn.__class__.__name__
        if hasattr(fn, '__module__'):
            _f.__module__ = fn.__module__
        if hasattr(fn.__call__, '__doc__') and fn.__call__.__doc__:
            _f.__doc__ = fn.__call__.__doc__
        elif fn.__doc__:
            _f.__doc__ = fn.__doc__
        return _f