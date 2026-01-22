from __future__ import annotations
import collections.abc as collections_abc
import inspect
import itertools
import operator
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import cache_key as _cache_key
from . import coercions
from . import elements
from . import roles
from . import schema
from . import visitors
from .base import _clone
from .base import Executable
from .base import Options
from .cache_key import CacheConst
from .operators import ColumnOperators
from .. import exc
from .. import inspection
from .. import util
from ..util.typing import Literal
def _add_getter(self, key, getter_fn):
    bind_paths = object.__getattribute__(self, '_bind_paths')
    bind_path_key = (key, getter_fn)
    if bind_path_key in bind_paths:
        return bind_paths[bind_path_key]
    getter = getter_fn(key)
    elem = object.__getattribute__(self, '_to_evaluate')
    value = getter(elem)
    rolled_down_value = AnalyzedCode._roll_down_to_literal(value)
    if coercions._deep_is_literal(rolled_down_value):
        wrapper = PyWrapper(self._sa_fn, key, value, getter=getter)
        bind_paths[bind_path_key] = wrapper
        return wrapper
    else:
        return value