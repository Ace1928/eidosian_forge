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
class _InfiniteRangeSetData(_SetData):
    """Data class for a infinite set.

    This Set implements an interface to an *infinite set* defined by one
    or more NumericRange objects.  As there are an infinite
    number of members, Infinite Range Sets are not iterable.

    """
    __slots__ = ('_ranges',)

    def __init__(self, component):
        _SetData.__init__(self, component=component)
        self._ranges = None

    def get(self, value, default=None):
        if value.__class__ is tuple and len(value) == 1:
            v = value[0]
            if any((v in r for r in self._ranges)):
                return v
        if any((value in r for r in self._ranges)):
            return value
        return default

    def isdiscrete(self):
        """Returns True if this set admits only discrete members"""
        return all((r.isdiscrete() for r in self.ranges()))

    @property
    def dimen(self):
        return 1

    @property
    def domain(self):
        return Reals

    def clear(self):
        self._ranges = ()

    def ranges(self):
        return iter(self._ranges)