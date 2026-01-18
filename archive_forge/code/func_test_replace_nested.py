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
def test_replace_nested(self):
    m = ConcreteModel()
    m.x = Param(mutable=True)
    m.y = Var([1, 2, 3])
    e = m.y[1] * m.y[2] * m.y[2] * m.y[3] == 0
    f = ReplacementWalker_ReplaceInternal().dfs_postorder_stack(e)
    assertExpressionsEqual(self, m.y[1] * m.y[2] * m.y[2] * m.y[3] == 0, e)
    assertExpressionsEqual(self, m.y[1] + m.y[2] + m.y[2] + m.y[3] == 0, f)
    self.assertIs(type(f.arg(0)), LinearExpression)
    self.assertEqual(f.arg(0).nargs(), 4)