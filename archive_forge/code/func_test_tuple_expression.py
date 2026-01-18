import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr import (
from pyomo.core.base.constraint import _GeneralConstraintData
def test_tuple_expression(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.p = Param(mutable=True, initialize=0)
    m.c = Constraint()
    m.c = (m.x, m.y)
    self.assertTrue(m.c.equality)
    self.assertIs(type(m.c.expr), EqualityExpression)
    with self.assertRaisesRegex(ValueError, "Constraint 'c' does not have a proper value. Equality Constraints expressed as 2-tuples cannot contain None"):
        m.c = (m.x, None)
    with self.assertRaisesRegex(ValueError, "Constraint 'c' created with an invalid non-finite lower bound \\(inf\\)"):
        m.c = (m.x, float('inf'))
    with self.assertRaisesRegex(ValueError, "Equality constraint 'c' defined with non-finite term"):
        m.c = EqualityExpression((m.x, None))