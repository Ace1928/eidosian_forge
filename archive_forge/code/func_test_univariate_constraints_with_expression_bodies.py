from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_univariate_constraints_with_expression_bodies(self):
    m = self.create_nested_structure()
    m.x = Var(bounds=(-100, 102))
    m.outer_d1.c = Constraint(expr=-20 <= 2 * m.x)
    m.outer_d1.c2 = Constraint(expr=m.x - 1 <= 10)
    m.outer_d1.inner_d1.c = Constraint(expr=3 * m.x - 7 <= 2)
    m.outer_d1.inner_d2.c = Constraint(expr=m.x >= -7)
    m.outer_d2.c = Constraint(expr=m.x + 4 == 4)
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m)
    self.check_nested_model_disjunction(m, bt)
    self.assertFalse(m.outer_d1.c.active)
    self.assertFalse(m.outer_d1.c2.active)
    self.assertFalse(m.outer_d1.inner_d1.c.active)
    self.assertFalse(m.outer_d1.inner_d2.c.active)
    self.assertFalse(m.outer_d2.c.active)
    self.assertEqual(len(list(m.component_data_objects(Constraint, descend_into=(Block, Disjunct), active=True))), 4)