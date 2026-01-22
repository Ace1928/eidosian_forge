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
class TestIdentifyParams(unittest.TestCase):

    def test_identify_params_numeric(self):
        self.assertEqual(list(identify_mutable_parameters(5)), [])

    def test_identify_mutable_parameters(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.a = Var(initialize=1)
        m.b = Var(m.I, initialize=1)
        self.assertEqual(list(identify_mutable_parameters(m.a)), [])
        self.assertEqual(list(identify_mutable_parameters(m.b[1])), [])
        self.assertEqual(list(identify_mutable_parameters(m.a + m.b[1])), [])
        self.assertEqual(list(identify_mutable_parameters(m.a ** m.b[1])), [])
        self.assertEqual(list(identify_mutable_parameters(m.a ** m.b[1] + m.b[2])), [])
        self.assertEqual(list(identify_mutable_parameters(m.a ** m.b[1] + m.b[2] * m.b[3] * m.b[2])), [])

    def test_identify_mutable_parameters_constants(self):
        m = ConcreteModel()
        m.x = Var(initialize=1)
        m.x.fix()
        m.p = Param(initialize=2, mutable=False)
        m.p_m = Param(initialize=3, mutable=True)
        e1 = m.x + m.p + NumericConstant(5)
        self.assertEqual(list(identify_mutable_parameters(e1)), [])
        e2 = 5 * m.x + NumericConstant(3) * m.p_m + m.p == 0
        mut_params = list(identify_mutable_parameters(e2))
        self.assertEqual(len(mut_params), 1)
        self.assertIs(mut_params[0], m.p_m)

    def test_identify_duplicate_params(self):
        m = ConcreteModel()
        m.a = Param(initialize=1, mutable=True)
        self.assertEqual(list(identify_mutable_parameters(2 * m.a + 2 * m.a)), [m.a])

    def test_identify_mutable_parameters_expr(self):
        m = ConcreteModel()
        m.a = Param(initialize=1, mutable=True)
        m.b = Param(initialize=2, mutable=True)
        m.e = Expression(expr=3 * m.a)
        m.E = Expression([0, 1], initialize={0: 3 * m.a, 1: 4 * m.b})
        self.assertEqual(list(identify_mutable_parameters(m.b + m.e)), [m.b, m.a])
        self.assertEqual(list(identify_mutable_parameters(m.E[0])), [m.a])
        self.assertEqual(list(identify_mutable_parameters(m.E[1])), [m.b])

    def test_identify_mutable_parameters_logical_expr(self):
        m = ConcreteModel()
        m.a = Param(initialize=0, mutable=True)
        expr = m.a + 1 == 0
        param_set = ComponentSet(identify_mutable_parameters(expr))
        self.assertEqual(len(param_set), 1)
        self.assertIn(m.a, param_set)

    def test_identify_mutable_parameters_params(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.a = Param(initialize=1, mutable=True)
        m.b = Param(m.I, initialize=1, mutable=True)
        m.p = Var(initialize=1)
        m.x = ExternalFunction(library='foo.so', function='bar')
        self.assertEqual(list(identify_mutable_parameters(m.a)), [m.a])
        self.assertEqual(list(identify_mutable_parameters(m.b[1])), [m.b[1]])
        self.assertEqual(list(identify_mutable_parameters(m.a + m.b[1])), [m.a, m.b[1]])
        self.assertEqual(list(identify_mutable_parameters(m.a ** m.b[1])), [m.a, m.b[1]])
        self.assertEqual(list(identify_mutable_parameters(m.a ** m.b[1] + m.b[2])), [m.b[2], m.a, m.b[1]])
        self.assertEqual(list(identify_mutable_parameters(m.a ** m.b[1] + m.b[2] * m.b[3] * m.b[2])), [m.a, m.b[1], m.b[2], m.b[3]])
        self.assertEqual(list(identify_mutable_parameters(m.a ** m.b[1] + m.b[2] / m.b[3] * m.b[2])), [m.a, m.b[1], m.b[2], m.b[3]])
        self.assertEqual(list(identify_mutable_parameters(m.x(m.a, 'string_param', 1, []) * m.b[1])), [m.b[1], m.a])
        self.assertEqual(list(identify_mutable_parameters(m.x(m.p, 'string_param', 1, []) * m.b[1])), [m.b[1]])
        self.assertEqual(list(identify_mutable_parameters(tanh(m.a) * m.b[1])), [m.b[1], m.a])
        self.assertEqual(list(identify_mutable_parameters(abs(m.a) * m.b[1])), [m.b[1], m.a])
        self.assertEqual(list(identify_mutable_parameters(m.a ** m.a + m.a)), [m.a])