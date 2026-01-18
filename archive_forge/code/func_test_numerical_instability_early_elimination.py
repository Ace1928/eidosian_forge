import os
from os.path import abspath, dirname
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet
from pyomo.core import (
from pyomo.core.base import TransformationFactory
from pyomo.core.expr import log
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.gdp import Disjunction, Disjunct
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.opt import SolverFactory, check_available_solvers
import pyomo.contrib.fme.fourier_motzkin_elimination
from io import StringIO
import logging
import random
def test_numerical_instability_early_elimination(self):
    m = ConcreteModel()
    m.x = Var()
    m.x0 = Var()
    m.y = Var()
    m.cons1 = Constraint(expr=0 <= (4.27 + 1.123e-09) * m.x + 13 * m.y - m.x0)
    m.cons2 = Constraint(expr=m.x0 >= 12 * m.y + 4.27 * m.x)
    fme = TransformationFactory('contrib.fourier_motzkin_elimination')
    first = m.clone()
    second = m.clone()
    third = m.clone()
    fme.apply_to(first, vars_to_eliminate=[first.x0], zero_tolerance=1e-10)
    constraints = first._pyomo_contrib_fme_transformation.projected_constraints
    cons = constraints[1]
    self.assertEqual(cons.lower, 0)
    repn = generate_standard_repn(cons.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(repn.constant, 0)
    self.assertEqual(len(repn.linear_coefs), 2)
    self.assertIs(repn.linear_vars[0], first.x)
    self.assertAlmostEqual(repn.linear_coefs[0], 1.123e-09)
    self.assertIs(repn.linear_vars[1], first.y)
    self.assertEqual(repn.linear_coefs[1], 1)
    self.assertIsNone(cons.upper)
    fme.apply_to(second, vars_to_eliminate=[second.x0, second.x], zero_tolerance=1e-10)
    self.assertEqual(len(second._pyomo_contrib_fme_transformation.projected_constraints), 0)
    fme.apply_to(third, vars_to_eliminate=[third.x0], verbose=True, zero_tolerance=1e-08)
    constraints = third._pyomo_contrib_fme_transformation.projected_constraints
    cons = constraints[1]
    self.assertEqual(cons.lower, 0)
    self.assertIs(cons.body, third.y)
    self.assertIsNone(cons.upper)
    fme.apply_to(m, vars_to_eliminate=[m.x0, m.x], verbose=True, zero_tolerance=1e-08)
    constraints = m._pyomo_contrib_fme_transformation.projected_constraints
    cons = constraints[1]
    self.assertEqual(cons.lower, 0)
    self.assertIs(cons.body, m.y)
    self.assertIsNone(cons.upper)