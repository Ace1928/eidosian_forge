import os
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.opt.base import UnknownSolver
from pyomo.opt.base.solvers import SolverFactory
from pyomo.opt.solver import SystemCallSolver
def test_set_executable_isexe_abspath(self):
    with SystemCallSolver(type='test') as opt:
        self.assertEqual(id(opt._user_executable), id(None))
        opt.set_executable(isexe_abspath)
        self.assertEqual(opt._user_executable, isexe_abspath)
        self.assertEqual(opt.executable(), isexe_abspath)
        opt._user_executable = None
        opt.set_executable(isexe_abspath, validate=False)
        self.assertEqual(opt._user_executable, isexe_abspath)
        self.assertEqual(opt.executable(), isexe_abspath)