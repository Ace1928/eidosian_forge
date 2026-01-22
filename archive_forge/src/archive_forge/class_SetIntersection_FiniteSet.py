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
class SetIntersection_FiniteSet(_FiniteSetMixin, SetIntersection_InfiniteSet):
    __slots__ = tuple()

    def _iter_impl(self):
        set0, set1 = self._sets
        if not set0.isordered():
            if set1.isordered():
                set0, set1 = (set1, set0)
            elif not set0.isfinite():
                if set1.isfinite():
                    set0, set1 = (set1, set0)
                else:
                    ranges = []
                    for r0 in set0.ranges():
                        ranges.extend(r0.range_intersection(set1.ranges()))
                    return iter(RangeSet(ranges=ranges))
        return (s for s in set0 if s in set1)

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        return sum((1 for _ in self))