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
def test_simple_hull_example(self):
    m = ConcreteModel()
    m.x0 = Var(bounds=(0, 3))
    m.x1 = Var(bounds=(0, 3))
    m.x = Var(bounds=(0, 3))
    m.disaggregation = Constraint(expr=m.x == m.x0 + m.x1)
    m.y = Var(domain=Binary)
    m.cons = Constraint(expr=2 * m.y <= m.x1)
    fme = TransformationFactory('contrib.fourier_motzkin_elimination')
    fme.apply_to(m, vars_to_eliminate=[m.x0, m.x1])
    constraints = m._pyomo_contrib_fme_transformation.projected_constraints
    self.assertEqual(len(constraints), 1)
    cons = constraints[1]
    self.assertIsNone(cons.upper)
    self.assertEqual(value(cons.lower), 0)
    repn = generate_standard_repn(cons.body)
    self.assertEqual(repn.constant, 0)
    self.assertEqual(len(repn.linear_vars), 2)
    self.assertIs(repn.linear_vars[0], m.x)
    self.assertEqual(repn.linear_coefs[0], 1)
    self.assertIs(repn.linear_vars[1], m.y)
    self.assertEqual(repn.linear_coefs[1], -2)
    self.assertTrue(repn.is_linear())