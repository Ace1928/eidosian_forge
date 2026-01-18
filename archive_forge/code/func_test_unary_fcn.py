import logging
import math
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.compare import assertExpressionsStructurallyEqual
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.environ import ConcreteModel, Param, Var, ExternalFunction
def test_unary_fcn(self):
    m = ConcreteModel()
    m.x = Var(range(3), initialize=range(3))
    m.y = Var(initialize=5)
    m.p = Param(initialize=3, mutable=True)
    e = sin(2 * m.y)
    f = e.create_node_with_local_data(e.args)
    self.assertIsNot(f, e)
    self.assertIs(type(f), type(e))
    self.assertIs(f.args, e.args)
    self.assertIs(e._fcn, f._fcn)
    self.assertIs(e._name, f._name)
    f = e.create_node_with_local_data((m.x[1],))
    self.assertIsNot(f, e)
    self.assertIs(type(f), type(e))
    self.assertEqual(f.args, (m.x[1],))
    self.assertIs(e._fcn, f._fcn)
    self.assertIs(e._name, f._name)
    e = sin(2 * m.p)
    f = e.create_node_with_local_data(e.args)
    self.assertIsNot(f, e)
    self.assertIs(type(f), type(e))
    self.assertIs(e._fcn, f._fcn)
    self.assertIs(e._name, f._name)
    f = e.create_node_with_local_data((m.x[1],))
    self.assertIsNot(f, e)
    self.assertIs(type(f), UnaryFunctionExpression)
    self.assertEqual(f.args, (m.x[1],))
    self.assertIs(e._fcn, f._fcn)
    self.assertIs(e._name, f._name)