from pyomo.common.collections import ComponentSet
from pyomo.common.pyomo_typing import overload
from pyomo.contrib.cp.scheduling_expr.precedence_expressions import (
from pyomo.core import Integers, value
from pyomo.core.base import Any, ScalarVar, ScalarBooleanVar
from pyomo.core.base.block import _BlockData, Block
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.base.initializer import BoundInitializer, Initializer
from pyomo.core.expr import GetItemExpression
class ScalarIntervalVar(IntervalVarData, IntervalVar):

    def __init__(self, *args, **kwds):
        self._suppress_ctypes = set()
        IntervalVarData.__init__(self, self)
        IntervalVar.__init__(self, *args, **kwds)
        self._data[None] = self
        self._index = UnindexedComponent_index