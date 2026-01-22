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
class SetProduct_FiniteSet(_FiniteSetMixin, SetProduct_InfiniteSet):
    __slots__ = tuple()

    def _iter_impl(self):
        _iter = itertools.product(*self._sets)
        if FLATTEN_CROSS_PRODUCT and normalize_index.flatten and (self.dimen != len(self._sets)):
            return (self._flatten_product(_) for _ in _iter)
        return _iter

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        ans = 1
        for s in self._sets:
            ans *= max(0, len(s))
        return ans