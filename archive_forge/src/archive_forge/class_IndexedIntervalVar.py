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
class IndexedIntervalVar(IntervalVar):

    def __getitem__(self, args):
        tmp = args if args.__class__ is tuple else (args,)
        if any((hasattr(arg, 'is_potentially_variable') and arg.is_potentially_variable() for arg in tmp)):
            return GetItemExpression((self,) + tmp)
        return super().__getitem__(args)