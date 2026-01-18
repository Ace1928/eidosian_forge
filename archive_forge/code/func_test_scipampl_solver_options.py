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
def test_scipampl_solver_options(self):
    self.set_solvers(scip=None, fail=False)
    scip = SolverFactory('scip', solver_io='nl')
    m = self.model = ConcreteModel()
    m.v = Var()
    m.o = Objective(expr=m.v)
    m.c = Constraint(expr=m.v >= 1)
    scip._get_version()
    self.run.reset_mock()
    with self.assertRaises(FileNotFoundError) as cm:
        scip.solve(m, timelimit=10, options={'numerics/feastol': 1e-09})
    args = self.run.call_args[0][0]
    self.assertEqual(self.executable_paths['scipampl'], args[0])
    self.assertTrue(exists(args[1]))
    root, ext = splitext(args[1])
    self.assertEqual('.nl', ext)
    self.assertEqual(root + '.sol', cm.exception.filename)
    self.assertEqual('-AMPL', args[2])
    options_dir = self.run.call_args[1].get('cwd', None)
    if options_dir is not None and exists(options_dir + '/scip.set'):
        self.assertEqual(3, len(args))
        options_file = options_dir + '/scip.set'
    else:
        self.assertEqual(4, len(args))
        options_file = args[3]
        self.assertTrue(exists(options_file))
    with open(options_file, 'r') as options:
        lines = options.readlines()
        self.assertIn('numerics/feastol = 1e-09\n', lines)
        self.assertIn('limits/time = 10\n', lines)