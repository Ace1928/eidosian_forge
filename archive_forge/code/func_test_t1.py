import os
from os.path import abspath, dirname, normpath, join
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.repn.tests.lp_diff import load_and_compare_lp_baseline
from pyomo.scripting.util import cleanup
import pyomo.scripting.pyomo_main as main
def test_t1(self):
    self.problem = 'test_t1'
    self.run_bilevel(join(exdir, 't1.py'))
    self.check('t1', 'linear_dual')