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
class SetSymmetricDifference_FiniteSet(_FiniteSetMixin, SetSymmetricDifference_InfiniteSet):
    __slots__ = tuple()

    def _iter_impl(self):
        set0, set1 = self._sets
        return itertools.chain((_ for _ in set0 if _ not in set1), (_ for _ in set1 if _ not in set0))

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        return sum((1 for _ in self))