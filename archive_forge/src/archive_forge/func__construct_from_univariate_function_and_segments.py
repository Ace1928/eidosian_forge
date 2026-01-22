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
def _construct_from_univariate_function_and_segments(self, obj, func):
    for idx1, idx2 in obj._simplices:
        x1 = obj._points[idx1][0]
        x2 = obj._points[idx2][0]
        y = {x: func(x) for x in [x1, x2]}
        slope = (y[x2] - y[x1]) / (x2 - x1)
        intercept = y[x1] - slope * x1
        obj._linear_functions.append(_univariate_linear_functor(slope, intercept))
    return obj