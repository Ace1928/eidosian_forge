from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_transform_nested_model_no_0_terms(self):
    m = self.create_nested_model()
    m.outer_d2.c.deactivate()
    m.outer_d2.c2 = Constraint(expr=m.x == 101)
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m)
    cons = bt.get_transformed_constraints(m.x, m.outer)
    self.assertEqual(len(cons), 2)
    lb = cons[0]
    ub = cons[1]
    assertExpressionsEqual(self, lb.expr, -10.0 * m.outer_d1.binary_indicator_var + 101.0 * m.outer_d2.binary_indicator_var <= m.x)
    assertExpressionsEqual(self, ub.expr, 11.0 * m.outer_d1.binary_indicator_var + 101.0 * m.outer_d2.binary_indicator_var >= m.x)
    cons = bt.get_transformed_constraints(m.x, m.outer_d1.inner)
    self.assertEqual(len(cons), 2)
    lb = cons[0]
    ub = cons[1]
    assertExpressionsEqual(self, lb.expr, -10.0 * m.outer_d1.inner_d1.binary_indicator_var - 7.0 * m.outer_d1.inner_d2.binary_indicator_var <= m.x)
    assertExpressionsEqual(self, ub.expr, 3.0 * m.outer_d1.inner_d1.binary_indicator_var + 11.0 * m.outer_d1.inner_d2.binary_indicator_var >= m.x)
    self.assertIs(_parent_disjunct(lb), m.outer_d1)
    self.assertIs(_parent_disjunct(ub), m.outer_d1)
    self.assertFalse(m.outer_d1.c.active)
    self.assertFalse(m.outer_d1.inner_d1.c.active)
    self.assertFalse(m.outer_d1.inner_d2.c.active)
    self.assertFalse(m.outer_d2.c2.active)
    self.assertEqual(len(list(m.component_data_objects(Constraint, active=True, descend_into=(Block, Disjunct)))), 4)