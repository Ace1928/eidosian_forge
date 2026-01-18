import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
def test_results_format(self):
    opt = pyomo.opt.SolverFactory('stest1')
    opt._results_format = 'a'
    self.assertEqual(opt.results_format(), 'a')
    opt._results_format = None
    self.assertEqual(opt.results_format(), None)