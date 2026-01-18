import pyomo.environ as pyo
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.GAMS import GAMSShell, GAMSDirect, gdxcc_available
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output
import os, shutil
from tempfile import mkdtemp
@unittest.skipIf(not gamsgms_available, "The 'gams' executable is not available")
def test_options_gms(self):
    with SolverFactory('gams', solver_io='gms') as opt:
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=m.x >= 10)
        m.o = Objective(expr=m.x)
        opt.options['load_solutions'] = False
        opt.solve(m)
        self.assertEqual(m.x.value, None)
        opt.solve(m, load_solutions=True)
        self.assertEqual(m.x.value, 10)