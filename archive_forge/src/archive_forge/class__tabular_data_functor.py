import logging
from pyomo.common import DeveloperError
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import numpy as np
from pyomo.common.dependencies.scipy import spatial
from pyomo.contrib.piecewise.piecewise_linear_expression import (
from pyomo.core import Any, NonNegativeIntegers, value, Var
from pyomo.core.base.block import _BlockData, Block
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.expression import Expression
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import UnindexedComponent_set
from pyomo.core.base.initializer import Initializer
import pyomo.core.expr as EXPR
class _tabular_data_functor(AutoSlots.Mixin):
    __slots__ = ('tabular_data',)

    def __init__(self, tabular_data, tupleize=False):
        if not tupleize:
            self.tabular_data = tabular_data
        else:
            self.tabular_data = {(pt,): val for pt, val in tabular_data.items()}

    def __call__(self, *args):
        return self.tabular_data[args]