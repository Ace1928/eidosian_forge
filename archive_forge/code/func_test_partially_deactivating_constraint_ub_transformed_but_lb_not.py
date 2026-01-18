from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_partially_deactivating_constraint_ub_transformed_but_lb_not(self):
    m = ConcreteModel()
    m.w = Var()
    m.d = Disjunct([1, 2, 3])
    m.disjunction = Disjunction(expr=[m.d[1], m.d[2], m.d[3]])
    m.d[1].c = Constraint(expr=m.w == 45)
    m.d[2].c = Constraint(expr=m.w <= 36)
    m.d[3].c = Constraint(expr=m.w <= 232)
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m)
    cons = bt.get_transformed_constraints(m.w, m.disjunction)
    self.assertEqual(len(cons), 1)
    ub = cons[0]
    assertExpressionsEqual(self, ub.expr, 45.0 * m.d[1].binary_indicator_var + 36.0 * m.d[2].binary_indicator_var + 232.0 * m.d[3].binary_indicator_var >= m.w)
    self.assertFalse(m.d[1].c.active)
    self.assertFalse(m.d[2].c.active)
    self.assertFalse(m.d[3].c.active)
    c_lb = m.d[1].component('c_lb')
    self.assertIsInstance(c_lb, Constraint)
    self.assertTrue(c_lb.active)
    assertExpressionsEqual(self, c_lb.expr, m.w >= 45.0)
    self.assertEqual(len(list(m.component_data_objects(Constraint, active=True, descend_into=(Block, Disjunct)))), 2)