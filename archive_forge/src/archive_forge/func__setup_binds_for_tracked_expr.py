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
def _setup_binds_for_tracked_expr(self, expr):
    bindparam_lookup = {b.key: b for b in self._resolved_bindparams}

    def replace(element: Optional[visitors.ExternallyTraversible], **kw: Any) -> Optional[visitors.ExternallyTraversible]:
        if isinstance(element, elements.BindParameter):
            if element.key in bindparam_lookup:
                bind = bindparam_lookup[element.key]
                if element.expanding:
                    bind.expanding = True
                    bind.expand_op = element.expand_op
                    bind.type = element.type
                return bind
        return None
    if self._rec.is_sequence:
        expr = [visitors.replacement_traverse(sub_expr, {}, replace) for sub_expr in expr]
    elif getattr(expr, 'is_clause_element', False):
        expr = visitors.replacement_traverse(expr, {}, replace)
    return expr