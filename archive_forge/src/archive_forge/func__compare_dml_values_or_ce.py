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
def _compare_dml_values_or_ce(self, lv, rv, **kw):
    lvce = hasattr(lv, '__clause_element__')
    rvce = hasattr(rv, '__clause_element__')
    if lvce != rvce:
        return False
    elif lvce and (not self.compare_inner(lv, rv, **kw)):
        return False
    elif not lvce and lv != rv:
        return False
    elif not self.compare_inner(lv, rv, **kw):
        return False
    return True