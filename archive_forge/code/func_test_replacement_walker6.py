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
def test_replacement_walker6(self):
    M = ConcreteModel()
    M.x = Var()
    M.w = VarList()
    M.z = Param(mutable=True)
    e = M.z * 2 * 3
    walker = ReplacementWalkerTest3(M)
    f = walker.dfs_postorder_stack(e)
    self.assertTrue(not e.is_potentially_variable())
    self.assertTrue(f.is_potentially_variable())
    assertExpressionsEqual(self, M.z * 2 * 3, e)
    assertExpressionsEqual(self, ProductExpression([ProductExpression([2 * M.w[1], 2]), 3]), f)