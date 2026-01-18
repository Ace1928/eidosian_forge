import os
import platform
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import native_types, nonpyomo_leaf_types, NumericConstant
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import (
from pyomo.core.base.param import _ParamData, ScalarParam
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.core.expr.compare import assertExpressionsEqual
def test_replacement_walker3(self):
    M = ConcreteModel()
    M.x = Param(mutable=True)
    M.y = Var()
    M.w = VarList()
    e = sin(M.x) + M.x * M.y + 3 <= 0
    walker = ReplacementWalkerTest3(M)
    f = walker.dfs_postorder_stack(e)
    assertExpressionsEqual(self, sin(M.x) + M.x * M.y + 3 <= 0, e)
    assertExpressionsEqual(self, sin(2 * M.w[1]) + 2 * M.w[1] * M.y + 3 <= 0, f)