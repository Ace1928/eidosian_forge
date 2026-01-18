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
def test_pow_expr(self):
    m = ConcreteModel()
    m.x = Var()
    m.p = Param(mutable=True, initialize=1)
    e = m.x ** m.p
    cfg = VisitorConfig()
    repn = LinearRepnVisitor(*cfg).walk_expression(e)
    self.assertEqual(cfg.subexpr, {})
    self.assertEqual(cfg.var_map, {id(m.x): m.x})
    self.assertEqual(cfg.var_order, {id(m.x): 0})
    self.assertEqual(repn.multiplier, 1)
    self.assertEqual(repn.constant, 0)
    self.assertEqual(repn.linear, {id(m.x): 1})
    self.assertEqual(repn.nonlinear, None)
    m.p = 0
    cfg = VisitorConfig()
    repn = LinearRepnVisitor(*cfg).walk_expression(e)
    self.assertEqual(cfg.subexpr, {})
    self.assertEqual(cfg.var_map, {id(m.x): m.x})
    self.assertEqual(cfg.var_order, {id(m.x): 0})
    self.assertEqual(repn.multiplier, 1)
    self.assertEqual(repn.constant, 1)
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)
    m.p = 2
    cfg = VisitorConfig()
    repn = LinearRepnVisitor(*cfg).walk_expression(e)
    self.assertEqual(cfg.subexpr, {})
    self.assertEqual(cfg.var_map, {id(m.x): m.x})
    self.assertEqual(cfg.var_order, {id(m.x): 0})
    self.assertEqual(repn.multiplier, 1)
    self.assertEqual(repn.constant, 0)
    self.assertEqual(repn.linear, {})
    assertExpressionsEqual(self, repn.nonlinear, m.x ** 2)
    m.x.fix(2)
    cfg = VisitorConfig()
    repn = LinearRepnVisitor(*cfg).walk_expression(e)
    self.assertEqual(cfg.subexpr, {})
    self.assertEqual(cfg.var_map, {})
    self.assertEqual(cfg.var_order, {})
    self.assertEqual(repn.multiplier, 1)
    self.assertEqual(repn.constant, 4)
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)
    m.p = 1 / 2
    m.x = -1
    cfg = VisitorConfig()
    repn = LinearRepnVisitor(*cfg).walk_expression(e)
    self.assertEqual(cfg.subexpr, {})
    self.assertEqual(cfg.var_map, {})
    self.assertEqual(cfg.var_order, {})
    self.assertEqual(repn.multiplier, 1)
    self.assertStructuredAlmostEqual(repn.constant, InvalidNumber(1j))
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)
    m.x.unfix()
    e = (1 + m.x) ** 2
    cfg = VisitorConfig()
    visitor = LinearRepnVisitor(*cfg)
    visitor.max_exponential_expansion = 2
    repn = visitor.walk_expression(e)
    self.assertEqual(cfg.subexpr, {})
    self.assertEqual(cfg.var_map, {id(m.x): m.x})
    self.assertEqual(cfg.var_order, {id(m.x): 0})
    self.assertEqual(repn.multiplier, 1)
    self.assertEqual(repn.constant, 0)
    self.assertEqual(repn.linear, {})
    assertExpressionsEqual(self, repn.nonlinear, (m.x + 1) * (m.x + 1))
    cfg = VisitorConfig()
    visitor = LinearRepnVisitor(*cfg)
    visitor.max_exponential_expansion = 2
    visitor.expand_nonlinear_products = True
    repn = visitor.walk_expression(e)
    self.assertEqual(cfg.subexpr, {})
    self.assertEqual(cfg.var_map, {id(m.x): m.x})
    self.assertEqual(cfg.var_order, {id(m.x): 0})
    self.assertEqual(repn.multiplier, 1)
    self.assertEqual(repn.constant, 1)
    self.assertEqual(repn.linear, {id(m.x): 2})
    assertExpressionsEqual(self, repn.nonlinear, m.x * m.x)