from __future__ import annotations
from collections import deque
import collections.abc as collections_abc
import itertools
from itertools import zip_longest
import operator
import typing
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from . import operators
from .cache_key import HasCacheKey
from .visitors import _TraverseInternalsType
from .visitors import anon_map
from .visitors import ExternallyTraversible
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .. import util
from ..util import langhelpers
from ..util.typing import Self
def visit_string_multi_dict(self, attrname, left_parent, left, right_parent, right, **kw):
    for lk, rk in zip_longest(sorted(left.keys()), sorted(right.keys()), fillvalue=(None, None)):
        if lk != rk:
            return COMPARE_FAILED
        lv, rv = (left[lk], right[rk])
        lhc = isinstance(left, HasCacheKey)
        rhc = isinstance(right, HasCacheKey)
        if lhc and rhc:
            if lv._gen_cache_key(self.anon_map[0], []) != rv._gen_cache_key(self.anon_map[1], []):
                return COMPARE_FAILED
        elif lhc != rhc:
            return COMPARE_FAILED
        elif lv != rv:
            return COMPARE_FAILED