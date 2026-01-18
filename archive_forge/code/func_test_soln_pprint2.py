import json
import pickle
import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
from pyomo.common.dependencies import yaml, yaml_available
def test_soln_pprint2(self):
    """Write a solution with only zero values, using the Solution.pprint() method"""
    self.soln.variable[1]['Value'] = 0.0
    self.soln.variable[2]['Value'] = 0.0
    self.soln.variable[4]['Value'] = 0.0
    with open(join(currdir, 'soln_pprint2.out'), 'w') as f:
        f.write(str(self.soln))
    with open(join(currdir, 'soln_pprint2.out'), 'r') as f1, open(join(currdir, 'soln_pprint2.txt'), 'r') as f2:
        self.assertEqual(f1.read().strip(), f2.read().strip())