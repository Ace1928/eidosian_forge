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
class TestDirect_LinearExpression(unittest.TestCase):

    def test_LinearExpression_Param(self):
        m = ConcreteModel()
        N = 10
        S = list(range(1, N + 1))
        m.x = Var(S, initialize=lambda m, i: 1.0 / i)
        m.P = Param(S, initialize=lambda m, i: i)
        m.obj = Objective(expr=LinearExpression(constant=1.0, linear_coefs=[m.P[i] for i in S], linear_vars=[m.x[i] for i in S]))
        self.assertAlmostEqual(value(m.obj), N + 1)
        repn = generate_standard_repn(m.obj.expr)
        self.assertAlmostEqual(repn.constant, 1.0)
        self.assertTrue(len(repn.linear_coefs) == N)
        self.assertTrue(len(repn.linear_vars) == N)

    def test_LinearExpression_Number(self):
        m = ConcreteModel()
        N = 10
        S = list(range(1, N + 1))
        m.x = Var(S, initialize=lambda m, i: 1.0 / i)
        m.obj = Objective(expr=LinearExpression(constant=1.0, linear_coefs=[i for i in S], linear_vars=[m.x[i] for i in S]))
        self.assertAlmostEqual(value(m.obj), N + 1)
        repn = generate_standard_repn(m.obj.expr)
        self.assertAlmostEqual(repn.constant, 1.0)
        self.assertTrue(len(repn.linear_coefs) == N)
        self.assertTrue(len(repn.linear_vars) == N)

    def test_LinearExpression_MutableParam(self):
        m = ConcreteModel()
        N = 10
        S = list(range(1, N + 1))
        m.x = Var(S, initialize=lambda m, i: 1.0 / i)
        m.P = Param(S, initialize=lambda m, i: i, mutable=True)
        m.obj = Objective(expr=LinearExpression(constant=1.0, linear_coefs=[m.P[i] for i in S], linear_vars=[m.x[i] for i in S]))
        self.assertAlmostEqual(value(m.obj), N + 1)
        repn = generate_standard_repn(m.obj.expr)
        self.assertAlmostEqual(repn.constant, 1.0)
        self.assertTrue(len(repn.linear_coefs) == N)
        self.assertTrue(len(repn.linear_vars) == N)

    def test_LinearExpression_expression(self):
        m = ConcreteModel()
        N = 10
        S = list(range(1, N + 1))
        m.x = Var(S, initialize=lambda m, i: 1.0 / i)
        m.P = Param(S, initialize=lambda m, i: i, mutable=True)
        m.obj = Objective(expr=LinearExpression(constant=1.0, linear_coefs=[i * m.P[i] for i in S], linear_vars=[m.x[i] for i in S]))
        self.assertAlmostEqual(value(m.obj), sum((i for i in S)) + 1)
        repn = generate_standard_repn(m.obj.expr)
        self.assertAlmostEqual(repn.constant, 1.0)
        self.assertTrue(len(repn.linear_coefs) == N)
        self.assertTrue(len(repn.linear_vars) == N)

    def test_LinearExpression_polynomial_degree(self):
        m = ConcreteModel()
        m.S = RangeSet(2)
        m.var_1 = Var(initialize=0)
        m.var_2 = Var(initialize=0)
        m.var_3 = Var(m.S, initialize=0)

        def con_rule(model):
            return model.var_1 - (model.var_2 + sum_product(defaultdict(lambda: 6), model.var_3)) <= 0
        m.c1 = Constraint(rule=con_rule)
        m.var_1.fix(1)
        m.var_2.fix(1)
        m.var_3.fix(1)
        self.assertTrue(is_fixed(m.c1.body))
        self.assertEqual(polynomial_degree(m.c1.body), 0)

    def test_LinearExpression_is_fixed(self):
        m = ConcreteModel()
        m.S = RangeSet(2)
        m.var_1 = Var(initialize=0)
        m.var_2 = Var(initialize=0)
        m.var_3 = Var(m.S, initialize=0)

        def con_rule(model):
            return model.var_1 - (model.var_2 + sum_product(defaultdict(lambda: 6), model.var_3)) <= 0
        m.c1 = Constraint(rule=con_rule)
        m.var_1.fix(1)
        m.var_2.fix(1)
        self.assertFalse(is_fixed(m.c1.body))
        self.assertEqual(polynomial_degree(m.c1.body), 1)