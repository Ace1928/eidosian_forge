from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.expr.numeric_expr import (
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.util import InvalidNumber
from pyomo.environ import ConcreteModel, Var, Param, Any, log
def test_zero_elimination(self):
    m = ConcreteModel()
    m.x = Var(range(4))
    e = 0 * m.x[0] + 0 * m.x[1] * m.x[2] + 0 * log(m.x[3])
    cfg = VisitorConfig()
    repn = QuadraticRepnVisitor(*cfg).walk_expression(e)
    self.assertEqual(cfg.subexpr, {})
    self.assertEqual(cfg.var_map, {id(m.x[0]): m.x[0], id(m.x[1]): m.x[1], id(m.x[2]): m.x[2], id(m.x[3]): m.x[3]})
    self.assertEqual(cfg.var_order, {id(m.x[0]): 0, id(m.x[1]): 1, id(m.x[2]): 2, id(m.x[3]): 3})
    self.assertEqual(repn.multiplier, 1)
    self.assertEqual(repn.constant, 0)
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.quadratic, None)
    self.assertEqual(repn.nonlinear, None)
    m.p = Param(mutable=True, within=Any, initialize=None)
    e = m.p * m.x[0] + m.p * m.x[1] * m.x[2] + m.p * log(m.x[3])
    cfg = VisitorConfig()
    repn = QuadraticRepnVisitor(*cfg).walk_expression(e)
    self.assertEqual(cfg.subexpr, {})
    self.assertEqual(cfg.var_map, {id(m.x[0]): m.x[0], id(m.x[1]): m.x[1], id(m.x[2]): m.x[2], id(m.x[3]): m.x[3]})
    self.assertEqual(cfg.var_order, {id(m.x[0]): 0, id(m.x[1]): 1, id(m.x[2]): 2, id(m.x[3]): 3})
    self.assertEqual(repn.multiplier, 1)
    self.assertEqual(repn.constant, 0)
    self.assertEqual(repn.linear, {id(m.x[0]): InvalidNumber(None)})
    self.assertEqual(repn.quadratic, {(id(m.x[1]), id(m.x[2])): InvalidNumber(None)})
    self.assertEqual(repn.nonlinear, InvalidNumber(None))