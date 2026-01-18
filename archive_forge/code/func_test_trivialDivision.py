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
def test_trivialDivision(self):
    m = AbstractModel()
    m.a = Var()
    m.p = Param()
    m.q = Param(initialize=2)
    m.r = Param(mutable=True)
    self.assertRaises(ZeroDivisionError, m.a.__div__, 0)
    e = 0 / m.a
    self.assertExpressionsEqual(e, DivisionExpression((0, m.a)))
    e = m.a / 1
    self.assertExpressionsEqual(e, m.a)
    e = 1 / m.a
    self.assertExpressionsEqual(e, DivisionExpression((1, m.a)))
    e = 1 / m.p
    self.assertIs(type(e), NPV_DivisionExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertEqual(e.arg(0), 1)
    self.assertIs(e.arg(1), m.p)
    e = 1 / m.q
    self.assertIs(type(e), NPV_DivisionExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertEqual(e.arg(0), 1)
    self.assertIs(e.arg(1), m.q)
    e = 1 / m.r
    self.assertIs(type(e), NPV_DivisionExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertEqual(e.arg(0), 1)
    self.assertIs(e.arg(1), m.r)
    e = NumericConstant(3) / NumericConstant(2)
    self.assertIs(type(e), float)
    self.assertEqual(e, 1.5)