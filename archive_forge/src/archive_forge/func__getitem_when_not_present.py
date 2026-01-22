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
def _getitem_when_not_present(self, index):
    if index is None and (not self.is_indexed()):
        obj = self._data[index] = self
    else:
        obj = self._data[index] = self._ComponentDataClass(component=self)
    obj._index = index
    parent = obj.parent_block()
    nonlinear_function = None
    if self._func_rule is not None:
        nonlinear_function = self._func_rule(parent, index)
    elif self._func is not None:
        nonlinear_function = self._func
    handler = self._handlers.get((nonlinear_function is not None, self._points_rule is not None, self._simplices_rule is not None, self._linear_funcs_rule is not None, self._tabular_data is not None or self._tabular_data_rule is not None))
    if handler is None:
        raise ValueError('Unsupported set of arguments given for constructing PiecewiseLinearFunction. Expected a nonlinear function and a listof breakpoints, a nonlinear function and a list of simplices, a list of linear functions and a list of corresponding simplices, or a dictionary mapping points to nonlinear function values.')
    return handler(self, obj, parent, nonlinear_function)