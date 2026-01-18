from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_tightest_bound_is_at_root(self):
    """
        x >= 60
        [ [x >= 55, [ ] v [x >= 66] ] ] v [x >= 5]
        """
    m = ConcreteModel()
    m.x = Var()
    m.x.setlb(4)
    m.c = Constraint(expr=m.x >= 60)
    m.d = Disjunct([1, 2])
    m.inner1 = Disjunct([1, 2])
    m.inner2 = Disjunct([1, 2])
    m.disjunction = Disjunction(expr=[m.d[1], m.d[2]])
    m.d[1].disjunction = Disjunction(expr=[m.inner1[1], m.inner1[2]])
    m.inner1[1].disjunction = Disjunction(expr=[m.inner2[1], m.inner2[2]])
    m.d[2].c = Constraint(expr=m.x >= 5)
    m.inner1[1].c = Constraint(expr=m.x >= 55)
    m.inner2[2].c = Constraint(expr=m.x >= 66)
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m)
    cons = bt.get_transformed_constraints(m.x, m.disjunction)
    self.assertEqual(len(cons), 1)
    lb = cons[0]
    print(lb.expr)
    assertExpressionsEqual(self, lb.expr, 60.0 * m.d[1].binary_indicator_var + 60.0 * m.d[2].binary_indicator_var <= m.x)
    self.assertIsNone(_parent_disjunct(lb))
    cons = bt.get_transformed_constraints(m.x, m.d[1].disjunction)
    self.assertEqual(len(cons), 1)
    lb = cons[0]
    assertExpressionsEqual(self, lb.expr, 60.0 * m.inner1[1].binary_indicator_var + 60.0 * m.inner1[2].binary_indicator_var <= m.x)
    self.assertIs(_parent_disjunct(lb), m.d[1])
    cons = bt.get_transformed_constraints(m.x, m.inner1[1].disjunction)
    self.assertEqual(len(cons), 1)
    lb = cons[0]
    assertExpressionsEqual(self, lb.expr, 60.0 * m.inner2[1].binary_indicator_var + 66.0 * m.inner2[2].binary_indicator_var <= m.x)
    self.assertIs(_parent_disjunct(lb), m.inner1[1])
    self.assertEqual(len(list(m.component_data_objects(Constraint, descend_into=(Block, Disjunct), active=True))), 4)