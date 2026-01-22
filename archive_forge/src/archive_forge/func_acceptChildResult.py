import ctypes
import logging
import os
from pyomo.common.fileutils import Library
from pyomo.core import value, Expression
from pyomo.core.base.block import SubclassOf
from pyomo.core.base.expression import _ExpressionData
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, identify_variables
from pyomo.common.collections import ComponentMap
def acceptChildResult(self, node, data, child_result, child_idx):
    self.refs.add(child_result)
    data.append(child_result)
    return data