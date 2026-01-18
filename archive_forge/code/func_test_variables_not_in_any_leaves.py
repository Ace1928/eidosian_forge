from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_variables_not_in_any_leaves(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.disjunct1 = Disjunct()
    m.disjunct1.c = Constraint(expr=m.x <= 9.7)
    m.disjunct1.disjunct1 = Disjunct()
    m.disjunct1.disjunct1.c = Constraint(expr=m.x + m.y <= 4)
    m.disjunct1.disjunct2 = Disjunct()
    m.disjunct1.disjunct2.c = Constraint(expr=m.y <= 9)
    m.disjunct1.disjunction = Disjunction(expr=[m.disjunct1.disjunct1, m.disjunct1.disjunct2])
    m.disjunct2 = Disjunct()
    m.disjunct2.c = Constraint(expr=m.x <= 9)
    m.disjunction = Disjunction(expr=[m.disjunct1, m.disjunct2])
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m)
    cons = bt.get_transformed_constraints(m.x, m.disjunction)
    self.assertEqual(len(cons), 1)
    ub = cons[0]
    assertExpressionsEqual(self, ub.expr, 9.7 * m.disjunct1.binary_indicator_var + 9.0 * m.disjunct2.binary_indicator_var >= m.x)
    cons = bt.get_transformed_constraints(m.y, m.disjunction)
    self.assertEqual(len(cons), 0)
    self.assertFalse(m.disjunct1.c.active)
    self.assertFalse(m.disjunct2.c.active)
    self.assertEqual(len(list(m.component_data_objects(Constraint, active=True, descend_into=(Block, Disjunct)))), 3)