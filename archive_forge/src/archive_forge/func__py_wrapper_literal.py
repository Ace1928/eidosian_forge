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
def _py_wrapper_literal(self, expr=None, operator=None, **kw):
    param = object.__getattribute__(self, '_param')
    to_evaluate = object.__getattribute__(self, '_to_evaluate')
    if param is None:
        name = object.__getattribute__(self, '_name')
        self._param = param = elements.BindParameter(name, required=False, unique=True, _compared_to_operator=operator, _compared_to_type=expr.type if expr is not None else None)
        self._has_param = True
    return param._with_value(to_evaluate, maintain_key=True)