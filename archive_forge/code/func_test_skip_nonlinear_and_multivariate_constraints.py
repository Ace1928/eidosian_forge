from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_skip_nonlinear_and_multivariate_constraints(self):
    m = self.create_nested_model()
    m.y = Var()
    m.z = Var()
    m.outer_d1.nonlinear = Constraint(expr=m.y ** 2 <= 7)
    m.outer_d1.inner_d2.multivariate = Constraint(expr=m.x + m.y <= m.z)
    m.outer_d2.leave_it = Constraint(expr=m.z == 7)
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m)
    self.check_nested_model_disjunction(m, bt)
    self.assertTrue(m.outer_d1.nonlinear.active)
    self.assertTrue(m.outer_d1.inner_d2.multivariate.active)
    self.assertTrue(m.outer_d2.leave_it.active)
    self.assertFalse(m.outer_d1.c.active)
    self.assertFalse(m.outer_d1.inner_d1.c.active)
    self.assertFalse(m.outer_d1.inner_d2.c.active)
    self.assertFalse(m.outer_d2.c.active)
    self.assertEqual(len(list(m.component_data_objects(Constraint, descend_into=(Block, Disjunct), active=True))), 7)