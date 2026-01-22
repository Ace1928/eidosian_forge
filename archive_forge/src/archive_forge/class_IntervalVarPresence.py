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
class IntervalVarPresence(ScalarBooleanVar):
    """This class defines the abstract interface for a single Boolean variable
    denoting whether or not an IntervalVar is scheduled"""
    __slots__ = ()

    def __init__(self):
        super().__init__(ctype=IntervalVarPresence)

    def get_associated_interval_var(self):
        return self.parent_block()