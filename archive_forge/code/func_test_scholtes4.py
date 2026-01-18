import os
from os.path import abspath, dirname, normpath, join
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
import pyomo.opt
import pyomo.scripting.pyomo_main as pyomo_main
from pyomo.scripting.util import cleanup
def test_scholtes4(self):
    self.problem = 'test_scholtes4'
    self.run_solver(join(exdir, 'scholtes4.py'))
    self.check('scholtes4', self.solver)