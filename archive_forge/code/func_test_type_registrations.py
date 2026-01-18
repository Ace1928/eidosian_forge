import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.expr.numeric_expr import LinearExpression, MonomialTermExpression
from pyomo.core.expr import Expr_if, inequality, LinearExpression, NPV_SumExpression
import pyomo.repn.linear as linear
from pyomo.repn.linear import LinearRepn, LinearRepnVisitor
from pyomo.repn.util import InvalidNumber
from pyomo.environ import (
def test_type_registrations(self):
    m = ConcreteModel()
    cfg = VisitorConfig()
    visitor = LinearRepnVisitor(*cfg)
    _orig_dispatcher = linear._before_child_dispatcher
    linear._before_child_dispatcher = bcd = _orig_dispatcher.__class__()
    bcd.clear()
    try:
        self.assertEqual(bcd.register_dispatcher(visitor, 5), (False, (linear._CONSTANT, 5)))
        self.assertEqual(len(bcd), 1)
        self.assertIs(bcd[int], bcd._before_native_numeric)
        self.assertEqual(bcd.register_dispatcher(visitor, 5j), (False, (linear._CONSTANT, 5j)))
        self.assertEqual(len(bcd), 2)
        self.assertIs(bcd[complex], bcd._before_complex)
        m.p = Param(initialize=5)
        self.assertEqual(bcd.register_dispatcher(visitor, m.p), (False, (linear._CONSTANT, 5)))
        self.assertEqual(len(bcd), 3)
        self.assertIs(bcd[m.p.__class__], bcd._before_param)
        m.q = Param([0], initialize=6, mutable=True)
        self.assertEqual(bcd.register_dispatcher(visitor, m.q[0]), (False, (linear._CONSTANT, 6)))
        self.assertEqual(len(bcd), 4)
        self.assertIs(bcd[m.q[0].__class__], bcd._before_param)
        self.assertEqual(bcd.register_dispatcher(visitor, m.p + m.q[0]), (False, (linear._CONSTANT, 11)))
        self.assertEqual(len(bcd), 6)
        self.assertIs(bcd[NPV_SumExpression], bcd._before_npv)
        self.assertIs(bcd[LinearExpression], bcd._before_general_expression)
        m.e = Expression(expr=m.p + m.q[0])
        self.assertEqual(bcd.register_dispatcher(visitor, m.e), (True, None))
        self.assertEqual(len(bcd), 7)
        self.assertIs(bcd[m.e.__class__], bcd._before_named_expression)
    finally:
        linear._before_child_dispatcher = _orig_dispatcher