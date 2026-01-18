import subprocess
import sys
from os.path import join, exists, splitext
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
import pyomo.environ
from pyomo.opt import SolverFactory
from pyomo.core import ConcreteModel, Var, Objective, Constraint
import pyomo.solvers.plugins.solvers.SCIPAMPL
def test_scip_solver_options(self):
    self.set_solvers(fail=False)
    scip = SolverFactory('scip', solver_io='nl')
    m = self.model = ConcreteModel()
    m.v = Var()
    m.o = Objective(expr=m.v)
    m.c = Constraint(expr=m.v >= 1)
    scip._get_version()
    self.run.reset_mock()
    with self.assertRaises(FileNotFoundError) as cm:
        scip.solve(m, timelimit=10)
    args = self.run.call_args[0][0]
    self.assertEqual(3, len(args))
    self.assertEqual(self.executable_paths['scip'], args[0])
    self.assertTrue(exists(args[1] + '.nl'))
    self.assertEqual(args[1] + '.sol', cm.exception.filename)
    self.assertEqual('-AMPL', args[2])
    options_dir = self.run.call_args[1]['cwd']
    self.assertTrue(exists(options_dir + '/scip.set'))
    with open(options_dir + '/scip.set', 'r') as options:
        self.assertEqual(['limits/time = 10\n'], options.readlines())