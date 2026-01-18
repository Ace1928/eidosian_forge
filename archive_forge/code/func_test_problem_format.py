import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
def test_problem_format(self):
    opt = pyomo.opt.SolverFactory('stest1')
    opt._problem_format = 'a'
    self.assertEqual(opt.problem_format(), 'a')
    opt._problem_format = None
    self.assertEqual(opt.problem_format(), None)