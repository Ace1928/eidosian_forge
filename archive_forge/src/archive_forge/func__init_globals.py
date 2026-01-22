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
def _init_globals(self, fn):
    build_py_wrappers = self.build_py_wrappers
    bindparam_trackers = self.bindparam_trackers
    track_bound_values = self.track_bound_values
    for name in fn.__code__.co_names:
        if name not in fn.__globals__:
            continue
        _bound_value = self._roll_down_to_literal(fn.__globals__[name])
        if coercions._deep_is_literal(_bound_value):
            build_py_wrappers.append((name, None))
            if track_bound_values:
                bindparam_trackers.append(self._bound_parameter_getter_func_globals(name))