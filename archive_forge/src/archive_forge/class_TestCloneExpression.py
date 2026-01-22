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
class TestCloneExpression(unittest.TestCase):

    def setUp(self):
        self.m = ConcreteModel()
        self.m.a = Var(initialize=5)
        self.m.b = Var(initialize=10)
        self.m.p = Param(initialize=1, mutable=True)

    def tearDown(self):
        self.m = None

    def test_numeric(self):
        with clone_counter() as counter:
            start = counter.count
            e_ = 1
            e = clone_expression(e_)
            self.assertEqual(id(e), id(e_))
            e = clone_expression(self.m.p)
            self.assertEqual(id(e), id(self.m.p))
            total = counter.count - start
            self.assertEqual(total, 2)

    def test_Expression(self):
        m = ConcreteModel()
        m.a = Var(initialize=1)
        m.b = Var(initialize=2)
        m.e = Expression(expr=3 * m.a)
        m.E = Expression([0, 1], initialize={0: 3 * m.a, 1: 4 * m.b})
        with clone_counter() as counter:
            start = counter.count
            expr1 = m.e + m.E[1]
            expr2 = expr1.clone()
            self.assertEqual(expr1(), 11)
            self.assertEqual(expr2(), 11)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            total = counter.count - start
            self.assertEqual(total, 1)

    def test_ExpressionX(self):
        m = ConcreteModel()
        m.a = Var(initialize=1)
        m.b = Var(initialize=2)
        m.e = Expression(expr=3 * m.a)
        m.E = Expression([0, 1], initialize={0: 3 * m.a, 1: 4 * m.b})
        with clone_counter() as counter:
            start = counter.count
            expr1 = m.e + m.E[1]
            expr2 = copy.deepcopy(expr1)
            self.assertEqual(expr1(), 11)
            self.assertEqual(expr2(), 11)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertNotEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            total = counter.count - start
            self.assertEqual(total, 0)

    def test_SumExpression(self):
        with clone_counter() as counter:
            start = counter.count
            expr1 = self.m.a + self.m.b
            expr2 = expr1.clone()
            self.assertEqual(expr1(), 15)
            self.assertEqual(expr2(), 15)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertIs(expr1.arg(0).arg(1), expr2.arg(0).arg(1))
            self.assertIs(expr1.arg(1).arg(1), expr2.arg(1).arg(1))
            expr1 += self.m.b
            self.assertEqual(expr1(), 25)
            self.assertEqual(expr2(), 15)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertIs(expr1.arg(0).arg(1), expr2.arg(0).arg(1))
            self.assertIs(expr1.arg(1).arg(1), expr2.arg(1).arg(1))
            total = counter.count - start
            self.assertEqual(total, 1)

    def test_SumExpressionX(self):
        with clone_counter() as counter:
            start = counter.count
            expr1 = self.m.a + self.m.b
            expr2 = copy.deepcopy(expr1)
            self.assertEqual(expr1(), 15)
            self.assertEqual(expr2(), 15)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertNotEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            expr1 += self.m.b
            self.assertEqual(expr1(), 25)
            self.assertEqual(expr2(), 15)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            total = counter.count - start
            self.assertEqual(total, 0)

    def test_SumExpressionY(self):
        self.m = ConcreteModel()
        A = range(5)
        self.m.a = Var(A, initialize=5)
        self.m.b = Var(initialize=10)
        with clone_counter() as counter:
            start = counter.count
            expr1 = quicksum((self.m.a[i] for i in self.m.a))
            expr2 = copy.deepcopy(expr1)
            self.assertEqual(expr1(), 25)
            self.assertEqual(expr2(), 25)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertNotEqual(id(expr1.linear_vars[0]), id(expr2.linear_vars[0]))
            self.assertNotEqual(id(expr1.linear_vars[1]), id(expr2.linear_vars[1]))
            expr1 += self.m.b
            self.assertEqual(expr1(), 35)
            self.assertEqual(expr2(), 25)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            total = counter.count - start
            self.assertEqual(total, 0)

    def test_ProductExpression_mult(self):
        with clone_counter() as counter:
            start = counter.count
            expr1 = self.m.a * self.m.b
            expr2 = expr1.clone()
            self.assertEqual(expr1(), 50)
            self.assertEqual(expr2(), 50)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertEqual(id(expr1._args_), id(expr2._args_))
            self.assertEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            expr1 *= self.m.b
            self.assertEqual(expr1(), 500)
            self.assertEqual(expr2(), 50)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertEqual(id(expr1.arg(0)._args_), id(expr2._args_))
            self.assertEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            self.assertEqual(id(expr1.arg(0).arg(0)), id(expr2.arg(0)))
            self.assertEqual(id(expr1.arg(0).arg(1)), id(expr2.arg(1)))
            expr1 = self.m.a * (self.m.b + self.m.a)
            expr2 = expr1.clone()
            self.assertEqual(expr1(), 75)
            self.assertEqual(expr2(), 75)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            total = counter.count - start
            self.assertEqual(total, 2)

    def test_ProductExpression_div(self):
        with clone_counter() as counter:
            start = counter.count
            expr1 = self.m.a / self.m.b
            expr2 = expr1.clone()
            self.assertEqual(expr1(), 0.5)
            self.assertEqual(expr2(), 0.5)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertEqual(id(expr1._args_), id(expr2._args_))
            self.assertEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            expr1 /= self.m.b
            self.assertEqual(expr1(), 0.05)
            self.assertEqual(expr2(), 0.5)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertEqual(id(expr1.arg(0)._args_), id(expr2._args_))
            self.assertEqual(id(expr1.arg(0).arg(0)), id(expr2.arg(0)))
            self.assertEqual(id(expr1.arg(0).arg(1)), id(expr2.arg(1)))
            expr1 = self.m.a / (self.m.b + self.m.a)
            expr2 = expr1.clone()
            self.assertEqual(expr1(), 1 / 3.0)
            self.assertEqual(expr2(), 1 / 3.0)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            total = counter.count - start
            self.assertEqual(total, 2)

    def test_sumOfExpressions(self):
        with clone_counter() as counter:
            start = counter.count
            expr1 = self.m.a * self.m.b + self.m.a * self.m.a
            expr2 = expr1.clone()
            self.assertEqual(expr1(), 75)
            self.assertEqual(expr2(), 75)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertEqual(expr1.arg(0)(), expr2.arg(0)())
            self.assertEqual(expr1.arg(1)(), expr2.arg(1)())
            self.assertNotEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            expr1 += self.m.b
            self.assertEqual(expr1(), 85)
            self.assertEqual(expr2(), 75)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertEqual(expr1.nargs(), 3)
            self.assertEqual(expr2.nargs(), 2)
            self.assertEqual(expr1.arg(0)(), 50)
            self.assertEqual(expr1.arg(1)(), 25)
            self.assertNotEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            total = counter.count - start
            self.assertEqual(total, 1)

    def test_productOfExpressions(self):
        with clone_counter() as counter:
            start = counter.count
            expr1 = (self.m.a + self.m.b) * (self.m.a + self.m.a)
            expr2 = expr1.clone()
            self.assertEqual(expr1(), 150)
            self.assertEqual(expr2(), 150)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertNotEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            self.assertEqual(expr1.arg(0)(), expr2.arg(0)())
            self.assertEqual(expr1.arg(1)(), expr2.arg(1)())
            self.assertEqual(expr1.arg(0).nargs(), 2)
            self.assertEqual(expr2.arg(0).nargs(), 2)
            self.assertEqual(expr1.arg(1).nargs(), 2)
            self.assertEqual(expr2.arg(1).nargs(), 2)
            self.assertIs(expr1.arg(0).arg(0).arg(1), expr2.arg(0).arg(0).arg(1))
            self.assertIs(expr1.arg(0).arg(1).arg(1), expr2.arg(0).arg(1).arg(1))
            self.assertIs(expr1.arg(1).arg(0).arg(1), expr2.arg(1).arg(0).arg(1))
            expr1 *= self.m.b
            self.assertEqual(expr1(), 1500)
            self.assertEqual(expr2(), 150)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            self.assertIs(type(expr1.arg(0)), type(expr2))
            self.assertEqual(expr1.arg(0)(), expr2())
            self.assertEqual(expr1.nargs(), 2)
            self.assertEqual(expr2.nargs(), 2)
            total = counter.count - start
            self.assertEqual(total, 1)

    def test_productOfExpressions_div(self):
        with clone_counter() as counter:
            start = counter.count
            expr1 = (self.m.a + self.m.b) / (self.m.a + self.m.a)
            expr2 = expr1.clone()
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertNotEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            self.assertEqual(expr1.arg(0)(), expr2.arg(0)())
            self.assertEqual(expr1.arg(1)(), expr2.arg(1)())
            self.assertEqual(expr1.nargs(), 2)
            self.assertEqual(expr2.nargs(), 2)
            self.assertEqual(expr1.arg(0).nargs(), 2)
            self.assertEqual(expr2.arg(0).nargs(), 2)
            self.assertEqual(expr1.arg(1).nargs(), 2)
            self.assertEqual(expr2.arg(1).nargs(), 2)
            self.assertIs(expr1.arg(0).arg(0).arg(1), expr2.arg(0).arg(0).arg(1))
            self.assertIs(expr1.arg(0).arg(1).arg(1), expr2.arg(0).arg(1).arg(1))
            expr1 /= self.m.b
            self.assertAlmostEqual(expr1(), 0.15)
            self.assertAlmostEqual(expr2(), 1.5)
            self.assertNotEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            self.assertIs(type(expr1.arg(0)), type(expr2))
            self.assertAlmostEqual(expr1.arg(0)(), expr2())
            self.assertEqual(expr1.nargs(), 2)
            self.assertEqual(expr2.nargs(), 2)
            total = counter.count - start
            self.assertEqual(total, 1)

    def test_Expr_if(self):
        with clone_counter() as counter:
            start = counter.count
            expr1 = Expr_if(IF=self.m.a + self.m.b < 20, THEN=self.m.a, ELSE=self.m.b)
            expr2 = expr1.clone()
            self.assertExpressionsStructurallyEqual(expr1, expr2)
            self.assertIsNot(expr1, expr2)
            self.assertIsNot(expr1.arg(0), expr2.arg(0))

    def test_LinearExpression(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1, 2])
        e = LinearExpression()
        f = e.clone()
        self.assertIsNot(e, f)
        self.assertIsNot(e.linear_coefs, f.linear_coefs)
        self.assertIsNot(e.linear_vars, f.linear_vars)
        self.assertEqual(e.constant, f.constant)
        self.assertEqual(e.linear_coefs, f.linear_coefs)
        self.assertEqual(e.linear_vars, f.linear_vars)
        self.assertEqual(f.constant, 0)
        self.assertEqual(f.linear_coefs, [])
        self.assertEqual(f.linear_vars, [])
        e = LinearExpression(constant=5, linear_vars=[m.x, m.y[1]], linear_coefs=[10, 20])
        f = e.clone()
        self.assertIsNot(e, f)
        self.assertIsNot(e.linear_coefs, f.linear_coefs)
        self.assertIsNot(e.linear_vars, f.linear_vars)
        self.assertEqual(e.constant, f.constant)
        self.assertEqual(e.linear_coefs, f.linear_coefs)
        self.assertEqual(e.linear_vars, f.linear_vars)
        self.assertEqual(f.constant, 5)
        self.assertEqual(f.linear_coefs, [10, 20])
        self.assertEqual(f.linear_vars, [m.x, m.y[1]])

    def test_getitem(self):
        with clone_counter() as counter:
            start = counter.count
            m = ConcreteModel()
            m.I = RangeSet(1, 9)
            m.x = Var(m.I, initialize=lambda m, i: i + 1)
            m.P = Param(m.I, initialize=lambda m, i: 10 - i, mutable=True)
            t = IndexTemplate(m.I)
            e = m.x[t + m.P[t + 1]] + 3
            e_ = e.clone()
            self.assertEqual('x[{I} + P[{I} + 1]] + 3', str(e_))
            total = counter.count - start
            self.assertEqual(total, 1)

    def test_other(self):
        with clone_counter() as counter:
            start = counter.count
            model = ConcreteModel()
            model.a = Var()
            model.x = ExternalFunction(library='foo.so', function='bar')
            e = model.x(2 * model.a, 1, 'foo', [])
            e_ = e.clone()
            self.assertEqual(type(e_), type(e))
            self.assertEqual(type(e_.arg(0)), type(e.arg(0)))
            self.assertEqual(type(e_.arg(1)), type(e.arg(1)))
            self.assertEqual(type(e_.arg(2)), type(e.arg(2)))
            self.assertEqual(type(e_.arg(3)), type(e.arg(3)))
            total = counter.count - start
            self.assertEqual(total, 1)

    def test_abs(self):
        with clone_counter() as counter:
            start = counter.count
            expr1 = abs(self.m.a)
            expr2 = expr1.clone()
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertEqual(expr1(), value(self.m.a))
            self.assertEqual(expr2(), value(self.m.a))
            self.assertEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            total = counter.count - start
            self.assertEqual(total, 1)

    def test_sin(self):
        with clone_counter() as counter:
            start = counter.count
            expr1 = sin(self.m.a)
            expr2 = expr1.clone()
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertEqual(expr1(), math.sin(value(self.m.a)))
            self.assertEqual(expr2(), math.sin(value(self.m.a)))
            self.assertEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            total = counter.count - start
            self.assertEqual(total, 1)