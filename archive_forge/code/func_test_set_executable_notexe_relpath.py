import os
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.opt.base import UnknownSolver
from pyomo.opt.base.solvers import SolverFactory
from pyomo.opt.solver import SystemCallSolver
@unittest.skipIf(is_windows, 'Skipping test because it requires testing if a file is executable on Windows')
def test_set_executable_notexe_relpath(self):
    with SystemCallSolver(type='test') as opt:
        self.assertEqual(id(opt._user_executable), id(None))
        with self.assertRaises(ValueError):
            opt.set_executable(notexe_relpath)
        self.assertEqual(id(opt._user_executable), id(None))
        opt.set_executable(notexe_relpath, validate=False)
        self.assertEqual(opt._user_executable, notexe_relpath)
        self.assertEqual(opt.executable(), notexe_relpath)