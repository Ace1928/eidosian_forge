import os
from os.path import abspath, dirname, normpath, join
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
import pyomo.opt
import pyomo.scripting.pyomo_main as pyomo_main
from pyomo.scripting.util import cleanup
@unittest.skipIf(not yaml_available, 'YAML is not available')
@unittest.skipIf(not 'ipopt' in solvers, "The 'ipopt' executable is not available")
class Solve_IPOPT(unittest.TestCase, CommonTests):

    def tearDown(self):
        if os.path.exists(os.path.join(currdir, 'result.yml')):
            os.remove(os.path.join(currdir, 'result.yml'))

    def run_solver(self, *args, **kwds):
        kwds['solver'] = 'ipopt'
        CommonTests.run_solver(self, *args, **kwds)