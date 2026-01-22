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
class _InsertionOrderSetData(_OrderedSetData):
    """
    This class defines the data for a ordered set where the items are ordered
    in insertion order (similar to Python's OrderedSet.

    Constructor Arguments:
        component   The Set object that owns this data.

    Public Class Attributes:
    """
    __slots__ = ()

    def set_value(self, val):
        if type(val) in Set._UnorderedInitializers:
            logger.warning('Calling set_value() on an insertion order Set with a fundamentally unordered data source (type: %s).  This WILL potentially lead to nondeterministic behavior in Pyomo' % (type(val).__name__,))
        super(_InsertionOrderSetData, self).set_value(val)

    def update(self, values):
        if type(values) in Set._UnorderedInitializers:
            logger.warning('Calling update() on an insertion order Set with a fundamentally unordered data source (type: %s).  This WILL potentially lead to nondeterministic behavior in Pyomo' % (type(values).__name__,))
        super(_InsertionOrderSetData, self).update(values)