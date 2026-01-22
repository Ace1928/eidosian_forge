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
@_define_handler(_handlers, False, False, True, True, False)
def _construct_from_linear_functions_and_simplices(self, obj, parent, nonlinear_function):
    obj._get_simplices_from_arg(self._simplices_rule(parent, obj._index))
    obj._linear_functions = [f for f in self._linear_funcs_rule(parent, obj._index)]
    return obj