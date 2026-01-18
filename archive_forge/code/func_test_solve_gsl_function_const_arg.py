import os
import shutil
import sys
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.gsl import find_GSL
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from pyomo.core.base.external import (
from pyomo.core.base.units_container import pint_available, units
from pyomo.core.expr.numeric_expr import (
from pyomo.opt import check_available_solvers
@unittest.skipIf(not check_available_solvers('ipopt'), "The 'ipopt' solver is not available")
def test_solve_gsl_function_const_arg(self):
    DLL = find_GSL()
    if not DLL:
        self.skipTest('Could not find the amplgsl.dll library')
    model = ConcreteModel()
    model.z_func = ExternalFunction(library=DLL, function='gsl_sf_beta')
    model.x = Var(initialize=1, bounds=(0.1, None))
    model.o = Objective(expr=-model.z_func(1, model.x))
    opt = SolverFactory('ipopt')
    res = opt.solve(model, tee=True)
    self.assertAlmostEqual(value(model.x), 0.1, 5)