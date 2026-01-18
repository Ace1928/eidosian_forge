import os
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.opt.base import UnknownSolver
from pyomo.opt.base.solvers import SolverFactory
from pyomo.opt.solver import SystemCallSolver
def test_set_executable_isexe_nopath(self):
    with SystemCallSolver(type='test') as opt:
        self.assertEqual(id(opt._user_executable), id(None))
        with self.assertRaises(ValueError):
            opt.set_executable(isexe_nopath)
        self.assertEqual(id(opt._user_executable), id(None))
        opt.set_executable(isexe_nopath, validate=False)
        self.assertEqual(opt._user_executable, isexe_nopath)
        self.assertEqual(opt.executable(), isexe_nopath)
        opt._user_executable = None
        rm_PATH = False
        orig_PATH = None
        if 'PATH' in os.environ:
            orig_PATH = os.environ['PATH']
        else:
            rm_PATH = True
            os.environ['PATH'] = ''
        os.environ['PATH'] = exedir + os.pathsep + os.environ['PATH']
        try:
            opt.set_executable(isexe_nopath)
            self.assertEqual(opt._user_executable, isexe_abspath)
            self.assertEqual(opt.executable(), isexe_abspath)
        finally:
            if rm_PATH:
                del os.environ['PATH']
            else:
                os.environ['PATH'] = orig_PATH