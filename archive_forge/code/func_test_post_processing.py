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
@unittest.skipIf(not 'glpk' in solvers, 'glpk not available')
def test_post_processing(self):
    m, disaggregatedVars = self.create_hull_model()
    fme = TransformationFactory('contrib.fourier_motzkin_elimination')
    fme.apply_to(m, vars_to_eliminate=disaggregatedVars, do_integer_arithmetic=True)
    fme.post_process_fme_constraints(m, SolverFactory('glpk'))
    constraints = m._pyomo_contrib_fme_transformation.projected_constraints
    self.assertEqual(len(constraints), 11)
    self.check_hull_projected_constraints(m, constraints, [8, 6, 20, 21, 13, 17, 9, 1, 2, 3, 4])
    for disj in m.component_data_objects(Disjunct):
        self.assertIs(disj.binary_indicator_var.domain, Binary)
    self.assertEqual(len([o for o in m.component_data_objects(Objective)]), 1)
    self.assertIsInstance(m.component('obj'), Objective)
    self.assertTrue(m.obj.active)