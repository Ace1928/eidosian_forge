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
def test_template_expr(self):
    m = ConcreteModel()
    m.I = RangeSet(1, 9)
    m.x = Var(m.I, initialize=lambda m, i: i + 1)
    m.P = Param(m.I, initialize=lambda m, i: 10 - i, mutable=True)
    t = IndexTemplate(m.I)
    e = m.x[t + m.P[t + 1]] + 3
    self.assertRaises(TemplateExpressionError, evaluate_expression, e)
    self.assertRaises(TemplateExpressionError, evaluate_expression, e, constant=True)