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
class SetDifference(SetOperator):
    __slots__ = tuple()
    _operator = ' - '

    def __new__(cls, *args):
        if cls != SetDifference:
            return super(SetDifference, cls).__new__(cls)
        set0, set1 = SetOperator._checkArgs(*args)
        if set0[0]:
            cls = SetDifference_OrderedSet
        elif set0[1]:
            cls = SetDifference_FiniteSet
        else:
            cls = SetDifference_InfiniteSet
        return cls.__new__(cls)

    def ranges(self):
        for a in self._sets[0].ranges():
            yield from a.range_difference(self._sets[1].ranges())

    @property
    def dimen(self):
        return self._sets[0].dimen