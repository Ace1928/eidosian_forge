import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr import (
from pyomo.core.base.constraint import _GeneralConstraintData
def test_potentially_variable_bounds(self):
    m = ConcreteModel()
    m.x = Var()
    m.l = Expression()
    m.u = Expression()
    m.c = Constraint(expr=inequality(m.l, m.x, m.u))
    self.assertIs(m.c.lower, m.l)
    self.assertIs(m.c.upper, m.u)
    with self.assertRaisesRegex(ValueError, 'No value for uninitialized NumericValue object l'):
        m.c.lb
    with self.assertRaisesRegex(ValueError, 'No value for uninitialized NumericValue object u'):
        m.c.ub
    m.l = 5
    m.u = 10
    self.assertIs(m.c.lower, m.l)
    self.assertIs(m.c.upper, m.u)
    self.assertEqual(m.c.lb, 5)
    self.assertEqual(m.c.ub, 10)
    m.l.expr = m.x
    with self.assertRaisesRegex(ValueError, "Constraint 'c' is a Ranged Inequality with a variable lower bound"):
        m.c.lower
    self.assertIs(m.c.upper, m.u)
    with self.assertRaisesRegex(ValueError, "Constraint 'c' is a Ranged Inequality with a variable lower bound"):
        m.c.lb
    self.assertEqual(m.c.ub, 10)
    m.l = 15
    m.u.expr = m.x
    self.assertIs(m.c.lower, m.l)
    with self.assertRaisesRegex(ValueError, "Constraint 'c' is a Ranged Inequality with a variable upper bound"):
        m.c.upper
    self.assertEqual(m.c.lb, 15)
    with self.assertRaisesRegex(ValueError, "Constraint 'c' is a Ranged Inequality with a variable upper bound"):
        m.c.ub
    m.l = -float('inf')
    m.u = float('inf')
    self.assertIs(m.c.lower, m.l)
    self.assertIs(m.c.upper, m.u)
    self.assertIsNone(m.c.lb)
    self.assertIsNone(m.c.ub)
    m.l = float('inf')
    m.u = -float('inf')
    self.assertIs(m.c.lower, m.l)
    self.assertIs(m.c.upper, m.u)
    with self.assertRaisesRegex(ValueError, "Constraint 'c' created with an invalid non-finite lower bound \\(inf\\)"):
        m.c.lb
    with self.assertRaisesRegex(ValueError, "Constraint 'c' created with an invalid non-finite upper bound \\(-inf\\)"):
        m.c.ub
    m.l = float('nan')
    m.u = -float('nan')
    self.assertIs(m.c.lower, m.l)
    self.assertIs(m.c.upper, m.u)
    with self.assertRaisesRegex(ValueError, "Constraint 'c' created with an invalid non-finite lower bound \\(nan\\)"):
        m.c.lb
    with self.assertRaisesRegex(ValueError, "Constraint 'c' created with an invalid non-finite upper bound \\(nan\\)"):
        m.c.ub