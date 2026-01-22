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
class TestExpression_Intrinsic(unittest.TestCase):

    def test_abs_numval(self):
        e = abs(1.5)
        self.assertAlmostEqual(value(e), 1.5)
        e = abs(-1.5)
        self.assertAlmostEqual(value(e), 1.5)

    def test_abs_param(self):
        m = ConcreteModel()
        m.p = Param(initialize=1.5)
        e = abs(m.p)
        self.assertAlmostEqual(value(e), 1.5)
        m.q = Param(initialize=-1.5)
        e = abs(m.q)
        self.assertAlmostEqual(value(e), 1.5)

    def test_abs_mutableparam(self):
        m = ConcreteModel()
        m.p = Param(initialize=0, mutable=True)
        m.p.value = 1.5
        e = abs(m.p)
        self.assertAlmostEqual(value(e), 1.5)
        m.p.value = -1.5
        e = abs(m.p)
        self.assertAlmostEqual(value(e), 1.5)
        self.assertIs(e.is_potentially_variable(), False)

    def test_ceil_numval(self):
        e = ceil(1.5)
        self.assertAlmostEqual(value(e), 2.0)
        e = ceil(-1.5)
        self.assertAlmostEqual(value(e), -1.0)

    def test_ceil_param(self):
        m = ConcreteModel()
        m.p = Param(initialize=1.5)
        e = ceil(m.p)
        self.assertAlmostEqual(value(e), 2.0)
        m.q = Param(initialize=-1.5)
        e = ceil(m.q)
        self.assertAlmostEqual(value(e), -1.0)

    def test_ceil_mutableparam(self):
        m = ConcreteModel()
        m.p = Param(initialize=0, mutable=True)
        m.p.value = 1.5
        e = ceil(m.p)
        self.assertAlmostEqual(value(e), 2.0)
        m.p.value = -1.5
        e = ceil(m.p)
        self.assertAlmostEqual(value(e), -1.0)
        self.assertIs(e.is_potentially_variable(), False)

    def test_ceil(self):
        m = ConcreteModel()
        m.v = Var()
        e = ceil(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 1.5
        self.assertAlmostEqual(value(e), 2.0)
        m.v.value = -1.5
        self.assertAlmostEqual(value(e), -1.0)
        self.assertIs(e.is_potentially_variable(), True)

    def test_floor(self):
        m = ConcreteModel()
        m.v = Var()
        e = floor(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 1.5
        self.assertAlmostEqual(value(e), 1.0)
        m.v.value = -1.5
        self.assertAlmostEqual(value(e), -2.0)

    def test_exp(self):
        m = ConcreteModel()
        m.v = Var()
        e = exp(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 1
        self.assertAlmostEqual(value(e), math.e)
        m.v.value = 0
        self.assertAlmostEqual(value(e), 1.0)

    def test_log(self):
        m = ConcreteModel()
        m.v = Var()
        e = log(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 1
        self.assertAlmostEqual(value(e), 0)
        m.v.value = math.e
        self.assertAlmostEqual(value(e), 1)

    def test_log10(self):
        m = ConcreteModel()
        m.v = Var()
        e = log10(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 1
        self.assertAlmostEqual(value(e), 0)
        m.v.value = 10
        self.assertAlmostEqual(value(e), 1)

    def test_pow(self):
        m = ConcreteModel()
        m.v = Var()
        m.p = Param(mutable=True)
        e = pow(m.v, m.p)
        self.assertEqual(e.__class__, PowExpression)
        m.v.value = 2
        m.p.value = 0
        self.assertAlmostEqual(value(e), 1.0)
        m.v.value = 2
        m.p.value = 1
        self.assertAlmostEqual(value(e), 2.0)

    def test_sqrt(self):
        m = ConcreteModel()
        m.v = Var()
        e = sqrt(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 1
        self.assertAlmostEqual(value(e), 1.0)
        m.v.value = 4
        self.assertAlmostEqual(value(e), 2.0)

    def test_sin(self):
        m = ConcreteModel()
        m.v = Var()
        e = sin(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = math.pi / 2.0
        self.assertAlmostEqual(value(e), 1.0)

    def test_cos(self):
        m = ConcreteModel()
        m.v = Var()
        e = cos(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0
        self.assertAlmostEqual(value(e), 1.0)
        m.v.value = math.pi / 2.0
        self.assertAlmostEqual(value(e), 0.0)

    def test_tan(self):
        m = ConcreteModel()
        m.v = Var()
        e = tan(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = math.pi / 4.0
        self.assertAlmostEqual(value(e), 1.0)

    def test_asin(self):
        m = ConcreteModel()
        m.v = Var()
        e = asin(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = 1.0
        self.assertAlmostEqual(value(e), math.pi / 2.0)

    def test_acos(self):
        m = ConcreteModel()
        m.v = Var()
        e = acos(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 1.0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = 0.0
        self.assertAlmostEqual(value(e), math.pi / 2.0)

    def test_atan(self):
        m = ConcreteModel()
        m.v = Var()
        e = atan(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = 1.0
        self.assertAlmostEqual(value(e), math.pi / 4.0)

    def test_sinh(self):
        m = ConcreteModel()
        m.v = Var()
        e = sinh(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0.0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = 1.0
        self.assertAlmostEqual(value(e), (math.e - 1.0 / math.e) / 2.0)

    def test_cosh(self):
        m = ConcreteModel()
        m.v = Var()
        e = cosh(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0.0
        self.assertAlmostEqual(value(e), 1.0)
        m.v.value = 1.0
        self.assertAlmostEqual(value(e), (math.e + 1.0 / math.e) / 2.0)

    def test_tanh(self):
        m = ConcreteModel()
        m.v = Var()
        e = tanh(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0.0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = 1.0
        self.assertAlmostEqual(value(e), (math.e - 1.0 / math.e) / (math.e + 1.0 / math.e))

    def test_asinh(self):
        m = ConcreteModel()
        m.v = Var()
        e = asinh(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0.0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = (math.e - 1.0 / math.e) / 2.0
        self.assertAlmostEqual(value(e), 1.0)

    def test_acosh(self):
        m = ConcreteModel()
        m.v = Var()
        e = acosh(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 1.0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = (math.e + 1.0 / math.e) / 2.0
        self.assertAlmostEqual(value(e), 1.0)

    def test_atanh(self):
        m = ConcreteModel()
        m.v = Var()
        e = atanh(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0.0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = (math.e - 1.0 / math.e) / (math.e + 1.0 / math.e)
        self.assertAlmostEqual(value(e), 1.0)