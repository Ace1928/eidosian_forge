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
def test_numerical_instability_almost_canceling(self):
    m = ConcreteModel()
    m.x = Var()
    m.x0 = Var()
    m.y = Var()
    m.cons1 = Constraint(expr=(1.342 + 2.371e-08) * m.x0 <= m.x + 17 * m.y)
    m.cons2 = Constraint(expr=(17.56 + 3.2e-07) * m.x0 >= m.y)
    fme = TransformationFactory('contrib.fourier_motzkin_elimination')
    fme.apply_to(m, vars_to_eliminate=[m.x0], verbose=True, zero_tolerance=1e-09)
    constraints = m._pyomo_contrib_fme_transformation.projected_constraints
    useful = constraints[1]
    repn = generate_standard_repn(useful.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_coefs), 2)
    self.assertEqual(useful.lower, 0)
    self.assertIs(repn.linear_vars[0], m.x)
    self.assertAlmostEqual(repn.linear_coefs[0], 0.7451564696962295)
    self.assertIs(repn.linear_vars[1], m.y)
    self.assertAlmostEqual(repn.linear_coefs[1], 12.610712377673217)
    self.assertEqual(repn.constant, 0)
    self.assertIsNone(useful.upper)