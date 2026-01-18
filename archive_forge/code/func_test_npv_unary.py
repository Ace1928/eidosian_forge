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
def test_npv_unary(self):
    m = ConcreteModel()
    m.p1 = Param(mutable=True)
    m.p2 = Param(mutable=True)
    m.x = Var(initialize=0)
    e1 = sin(m.p1)
    e2 = replace_expressions(e1, {id(m.p1): m.p2})
    e3 = replace_expressions(e1, {id(m.p1): m.x})
    assertExpressionsEqual(self, e2, sin(m.p2))
    assertExpressionsEqual(self, e3, sin(m.x))