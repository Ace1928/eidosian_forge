import inspect
import itertools
import logging
import math
import sys
import weakref
from pyomo.common.pyomo_typing import overload
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.errors import DeveloperError, PyomoException
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
from pyomo.core.base.range import (
from pyomo.core.base.component import (
from pyomo.core.base.indexed_component import (
from pyomo.core.base.global_set import (
from collections.abc import Sequence
from operator import itemgetter
class SetInitializer(InitializerBase):
    """An Initializer wrapper for returning Set objects

    This initializer wraps another Initializer and converts the return
    value to a proper Pyomo Set.  If the initializer is None, then Any
    is returned.  This initializer can be 'intersected' with another
    initializer to return the SetIntersect of the Sets returned by the
    initializers.

    """
    __slots__ = ('_set', 'verified')

    def __init__(self, init, allow_generators=True):
        self.verified = False
        if init is None:
            self._set = None
        else:
            self._set = Initializer(init, allow_generators=allow_generators, treat_sequences_as_mappings=False)

    def intersect(self, other):
        if self._set is None:
            if type(other) is SetInitializer:
                self._set = other._set
            else:
                self._set = other
        elif type(other) is SetInitializer:
            if other._set is not None:
                self._set = SetIntersectInitializer(self._set, other._set)
        else:
            self._set = SetIntersectInitializer(self._set, other)

    def __call__(self, parent, idx, obj):
        if self._set is None:
            return Any
        _ans, _anonymous = process_setarg(self._set(parent, idx))
        if _anonymous:
            pc = obj.parent_component()
            if getattr(pc, '_anonymous_sets', None) is None:
                pc._anonymous_sets = _anonymous
            else:
                pc._anonymous_sets.update(_anonymous)
            for _set in _anonymous:
                _set._parent = pc._parent
            if pc._constructed:
                for _set in _anonymous:
                    _set.construct()
        return _ans

    def constant(self):
        return self._set is None or self._set.constant()

    def contains_indices(self):
        return self._set is not None and self._set.contains_indices()

    def indices(self):
        if self._set is not None:
            return self._set.indices()
        else:
            super(SetInitializer, self).indices()

    def setdefault(self, val):
        if self._set is None:
            self._set = Initializer(val)