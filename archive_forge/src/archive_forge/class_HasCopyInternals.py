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
class HasCopyInternals(HasTraverseInternals):
    __slots__ = ()

    def _clone(self, **kw):
        raise NotImplementedError()

    def _copy_internals(self, *, omit_attrs: Iterable[str]=(), **kw: Any) -> None:
        """Reassign internal elements to be clones of themselves.

        Called during a copy-and-traverse operation on newly
        shallow-copied elements to create a deep copy.

        The given clone function should be used, which may be applying
        additional transformations to the element (i.e. replacement
        traversal, cloned traversal, annotations).

        """
        try:
            traverse_internals = self._traverse_internals
        except AttributeError:
            return
        for attrname, obj, meth in _copy_internals.run_generated_dispatch(self, traverse_internals, '_generated_copy_internals_traversal'):
            if attrname in omit_attrs:
                continue
            if obj is not None:
                result = meth(attrname, self, obj, **kw)
                if result is not None:
                    setattr(self, attrname, result)