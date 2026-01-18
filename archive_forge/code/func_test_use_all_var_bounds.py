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
def test_use_all_var_bounds(self):
    m = self.make_tiny_model_where_bounds_matter()
    fme = TransformationFactory('contrib.fourier_motzkin_elimination')
    fme.apply_to(m.b, vars_to_eliminate=[m.y])
    constraints = m.b._pyomo_contrib_fme_transformation.projected_constraints
    self.check_tiny_model_constraints(constraints)