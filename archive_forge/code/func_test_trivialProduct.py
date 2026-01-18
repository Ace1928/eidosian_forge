import copy
import pickle
import math
import os
from collections import defaultdict
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.environ import (
from pyomo.kernel import variable, expression, objective
from pyomo.core.expr.expr_common import ExpressionType, clone_counter
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import RelationalExpression, EqualityExpression
from pyomo.core.expr.relational_expr import RelationalExpression, EqualityExpression
from pyomo.common.errors import PyomoException
from pyomo.core.expr.visitor import expression_to_string, clone_expression
from pyomo.core.expr import Expr_if
from pyomo.core.base.label import NumericLabeler
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr import expr_common
from pyomo.core.base.var import _GeneralVarData
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numvalue import NumericValue
def test_trivialProduct(self):
    m = ConcreteModel()
    m.a = Var()
    m.p = Param(initialize=0)
    m.q = Param(initialize=1)
    e = m.a * 0
    self.assertExpressionsEqual(e, MonomialTermExpression((0, m.a)))
    e = 0 * m.a
    self.assertExpressionsEqual(e, MonomialTermExpression((0, m.a)))
    e = m.a * m.p
    self.assertExpressionsEqual(e, MonomialTermExpression((0, m.a)))
    e = m.p * m.a
    self.assertExpressionsEqual(e, MonomialTermExpression((0, m.a)))
    e = m.a * 1
    self.assertExpressionsEqual(e, m.a)
    e = 1 * m.a
    self.assertExpressionsEqual(e, m.a)
    e = m.a * m.q
    self.assertExpressionsEqual(e, m.a)
    e = m.q * m.a
    self.assertExpressionsEqual(e, m.a)
    e = NumericConstant(3) * NumericConstant(2)
    self.assertExpressionsEqual(e, 6)
    self.assertIs(type(e), int)
    self.assertEqual(e, 6)