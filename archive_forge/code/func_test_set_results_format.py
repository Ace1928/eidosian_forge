import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
def test_set_results_format(self):
    opt = pyomo.opt.SolverFactory('stest1')
    opt._valid_problem_formats = ['a']
    opt._valid_results_formats = {'a': 'b'}
    self.assertEqual(opt.problem_format(), None)
    try:
        opt.set_results_format('b')
    except ValueError:
        pass
    else:
        self.fail("Should not be able to set the results format unless it's declared as valid for the current problem format.")
    opt.set_problem_format('a')
    self.assertEqual(opt.problem_format(), 'a')
    opt.set_results_format('b')
    self.assertEqual(opt.results_format(), 'b')